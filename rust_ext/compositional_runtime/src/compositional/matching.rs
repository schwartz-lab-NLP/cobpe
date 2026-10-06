use super::*;

impl CompositionalTokenizer {
    pub(super) fn find_longest_boundary_safe_match(
        &self,
        raw_ids: &[u32],
        start_idx: usize,
        space_prefix_prefix_sum: &[usize],
    ) -> Option<EntryValue> {
        if start_idx >= raw_ids.len() {
            return None;
        }
        let start_inside_word = start_idx > 0
            && self.token_meta_ref(raw_ids[start_idx]).has_word_char
            && !self.token_meta_ref(raw_ids[start_idx]).has_space_prefix
            && self.token_meta_ref(raw_ids[start_idx - 1]).has_word_char;
        let mut node_idx = 0usize;
        let max_end = usize::min(start_idx + self.max_sequence_len, raw_ids.len());
        let mut best: Option<EntryValue> = None;
        let mut best_length = 0usize;
        for end_idx in start_idx..max_end {
            let token_id = raw_ids[end_idx];
            let child = match self.trie_nodes[node_idx].children.get(&token_id) {
                Some(idx) => *idx,
                None => break,
            };
            node_idx = child;
            let Some(entry) = self.trie_nodes[node_idx].value.clone() else {
                continue;
            };
            let span_end = end_idx + 1;
            let match_length = span_end - start_idx;
            let combined_modifier = self.combine_modifier_rows(&entry.modifier_rows);
            let allow_intra_word_cap_alias =
                self.is_intra_word_cap_alias_match(match_length, &combined_modifier);
            if start_inside_word && !allow_intra_word_cap_alias {
                continue;
            }
            if span_end < raw_ids.len()
                && self.token_meta_ref(raw_ids[end_idx]).has_word_char
                && !self.token_meta_ref(raw_ids[span_end]).has_space_prefix
                && self.token_meta_ref(raw_ids[span_end]).has_word_char
                && !allow_intra_word_cap_alias
            {
                continue;
            }
            if match_length > 1 {
                let all_word =
                    (start_idx..span_end).all(|j| self.token_meta_ref(raw_ids[j]).has_word_char);
                if all_word
                    && (space_prefix_prefix_sum[span_end] - space_prefix_prefix_sum[start_idx + 1])
                        > 0
                {
                    continue;
                }
            }
            best = Some(entry);
            best_length = match_length;
        }
        if best_length > 0 {
            best
        } else {
            None
        }
    }

    pub(super) fn should_prefer_cap_fallback_over_match(
        &self,
        raw_ids: &[u32],
        start_idx: usize,
        entry: &EntryValue,
    ) -> bool {
        let match_length = entry.consumed_len;
        let modifier = self.combine_modifier_rows(&entry.modifier_rows);
        if !self.is_intra_word_cap_alias_match(match_length, &modifier) {
            return false;
        }
        if start_idx >= raw_ids.len() || !self.token_meta_ref(raw_ids[start_idx]).has_word_char {
            return false;
        }
        let prev_continues_word = start_idx > 0
            && self.token_meta_ref(raw_ids[start_idx - 1]).has_word_char
            && !self.token_meta_ref(raw_ids[start_idx]).has_space_prefix;
        let next_idx = start_idx + match_length;
        let next_continues_word = next_idx < raw_ids.len()
            && self.token_meta_ref(raw_ids[next_idx]).has_word_char
            && !self.token_meta_ref(raw_ids[next_idx]).has_space_prefix;
        prev_continues_word || next_continues_word
    }

    pub(super) fn try_lowercase_cap_fallback(
        &self,
        raw_ids: &[u32],
        start_idx: usize,
        pending_groups: &[PendingGroup],
        pending_leading_space: bool,
    ) -> Option<(usize, Vec<u32>, Vec<Vec<u16>>)> {
        let base_cap_idx = self.group_idx("base_capitalization")?;
        if start_idx >= raw_ids.len() || !self.token_meta_ref(raw_ids[start_idx]).has_word_char {
            return None;
        }
        let mut end_idx = start_idx + 1;
        while end_idx < raw_ids.len() {
            let meta = self.token_meta_ref(raw_ids[end_idx]);
            if meta.has_space_prefix || !meta.has_word_char || meta.is_whitespace_only {
                break;
            }
            end_idx += 1;
        }
        let mut surface = self.decode_ids(&raw_ids[start_idx..end_idx]);
        surface = surface.trim().to_string();
        if surface.is_empty()
            || !surface.chars().all(|ch| ch.is_alphabetic())
            || !surface.chars().any(|ch| ch.is_uppercase())
        {
            return None;
        }
        if !is_base_cap_representable_surface(&surface) {
            return None;
        }
        let split_segments = split_camel_case_segments(&surface);
        let is_title_surface = surface
            .chars()
            .next()
            .map(|ch| ch.is_uppercase())
            .unwrap_or(false)
            && surface
                .chars()
                .skip(1)
                .all(|ch| !ch.is_alphabetic() || ch.is_lowercase());
        if split_segments.is_none() && !is_title_surface {
            return None;
        }
        let segments = expand_caps_segments(split_segments.unwrap_or_else(|| vec![surface]));
        let mut output_ids = Vec::new();
        let mut output_mods = Vec::new();
        let mut first_output = true;
        let space_idx = self.group_idx("space_prefix");
        for segment in segments {
            let lower_ids = self.encode_segment(&segment.to_lowercase())?;
            if lower_ids.is_empty() {
                return None;
            }
            let mut base_modifier = self.empty_modifier();
            base_modifier[base_cap_idx] = 1;
            if first_output {
                if pending_leading_space {
                    if let Some(idx) = space_idx {
                        base_modifier[idx] = 1;
                    }
                }
                base_modifier = self.combine_pending(&base_modifier, pending_groups);
            }
            let per_token_mods = self.spread_multi_token_modifiers(&base_modifier, lower_ids.len());
            output_ids.extend(lower_ids);
            output_mods.extend(per_token_mods);
            first_output = false;
        }
        Some((end_idx - start_idx, output_ids, output_mods))
    }

    pub(super) fn titlecase_lower_span(
        &self,
        raw_ids: &[u32],
        start_idx: usize,
    ) -> Option<(usize, String)> {
        if start_idx >= raw_ids.len() || !self.token_meta_ref(raw_ids[start_idx]).has_word_char {
            return None;
        }
        let mut end_idx = start_idx + 1;
        while end_idx < raw_ids.len() {
            let meta = self.token_meta_ref(raw_ids[end_idx]);
            if meta.has_space_prefix || !meta.has_word_char || meta.is_whitespace_only {
                break;
            }
            end_idx += 1;
        }
        let surface = self
            .decode_ids(&raw_ids[start_idx..end_idx])
            .trim()
            .to_string();
        if surface.is_empty()
            || !surface.chars().all(|ch| ch.is_alphabetic())
            || !surface.chars().any(|ch| ch.is_uppercase())
            || !is_base_cap_representable_surface(&surface)
        {
            return None;
        }
        let is_title_surface = surface
            .chars()
            .next()
            .map(|ch| ch.is_uppercase())
            .unwrap_or(false)
            && surface
                .chars()
                .skip(1)
                .all(|ch| !ch.is_alphabetic() || ch.is_lowercase());
        if !is_title_surface {
            return None;
        }
        Some((end_idx - start_idx, surface.to_lowercase()))
    }

    pub(super) fn can_attach_detached_modifier(
        &self,
        raw_ids: &[u32],
        start_idx: usize,
        consumed_len: usize,
        pending_has_prefix_punct: bool,
    ) -> bool {
        if consumed_len == 0 || start_idx >= raw_ids.len() {
            return false;
        }
        let span_end = usize::min(start_idx + consumed_len, raw_ids.len());
        // A detachable function word may begin after any whitespace boundary
        // (including a newline). The required lexical separator is the one
        // between the function word and its host, checked below.
        let left_ok = if start_idx == 0 {
            true
        } else {
            self.token_meta_ref(raw_ids[start_idx]).has_space_prefix
                || self
                    .token_meta_ref(raw_ids[start_idx - 1])
                    .is_whitespace_only
        };
        if !left_ok && !pending_has_prefix_punct {
            return false;
        }
        let Some((right_spaces, j)) = self.following_boundary_space_count(raw_ids, span_end) else {
            return false;
        };
        if right_spaces != 1 {
            return false;
        }
        if j >= raw_ids.len() || !self.raw_position_has_word_char(raw_ids, j) {
            return false;
        }
        let next_surface = self.word_surface(raw_ids, j, None);
        let current_surface = self.word_surface(raw_ids, start_idx, Some(span_end));
        let has_prep = self
            .runtime
            .literal_maps
            .get("prepositions")
            .map(|m| m.contains_key(&next_surface))
            .unwrap_or(false);
        if has_prep {
            return false;
        }
        let current_is_prep = self
            .runtime
            .literal_maps
            .get("prepositions")
            .map(|m| m.contains_key(&current_surface))
            .unwrap_or(false);
        let next_is_det = self
            .runtime
            .literal_maps
            .get("determiners")
            .map(|m| m.contains_key(&next_surface))
            .unwrap_or(false);
        let next_is_pronoun = self
            .runtime
            .literal_maps
            .get("pronouns")
            .map(|m| m.contains_key(&next_surface))
            .unwrap_or(false);
        if current_is_prep && next_is_det {
            return true;
        }
        if next_is_det {
            return false;
        }
        if next_is_pronoun {
            return false;
        }
        true
    }

    pub(super) fn postposition_after(
        &self,
        raw_ids: &[u32],
        host_last_idx: usize,
    ) -> Option<(usize, usize, LiteralTransform)> {
        let start_idx = host_last_idx.saturating_add(1);
        let Some((spaces, marker_idx)) = self.following_boundary_space_count(raw_ids, start_idx)
        else {
            return None;
        };
        if spaces != 1
            || marker_idx >= raw_ids.len()
            || !self.raw_position_has_word_char(raw_ids, marker_idx)
        {
            return None;
        }

        let first_word_end = self.word_span_end(raw_ids, marker_idx);
        let first_surface = self.word_surface(raw_ids, marker_idx, Some(first_word_end));
        let candidates = self.postpositions_by_first_word.get(&first_surface)?;
        let mut best: Option<(usize, LiteralTransform)> = None;
        for candidate in candidates {
            let mut candidate_end = first_word_end;
            let mut matched = true;
            for word in candidate.words.iter().skip(1) {
                let Some((separator_spaces, next_idx)) =
                    self.following_boundary_space_count(raw_ids, candidate_end)
                else {
                    matched = false;
                    break;
                };
                if separator_spaces != 1
                    || next_idx >= raw_ids.len()
                    || !self.raw_position_has_word_char(raw_ids, next_idx)
                {
                    matched = false;
                    break;
                }
                let word_idx = next_idx;
                let word_end = self.word_span_end(raw_ids, word_idx);
                if self.word_surface(raw_ids, word_idx, Some(word_end)) != *word {
                    matched = false;
                    break;
                }
                candidate_end = word_end;
            }
            if matched
                && best
                    .as_ref()
                    .map(|(end, _)| candidate_end > *end)
                    .unwrap_or(true)
            {
                best = Some((candidate_end, candidate.transform.clone()));
            }
        }
        let (marker_end, transform) = best?;
        if !self.raw_position_has_word_char(raw_ids, host_last_idx) {
            return None;
        }
        let mut host_start = host_last_idx;
        while host_start > 0 {
            let previous = self.token_meta_ref(raw_ids[host_start - 1]);
            if previous.is_whitespace_only
                || previous.has_space_prefix
                || !self.raw_position_has_word_char(raw_ids, host_start - 1)
            {
                break;
            }
            host_start -= 1;
        }
        let host_surface = self.word_surface(raw_ids, host_start, Some(host_last_idx + 1));
        if self
            .runtime
            .literal_maps
            .get("determiners")
            .map(|map| map.contains_key(&host_surface))
            .unwrap_or(false)
            || self
                .runtime
                .literal_maps
                .get("prepositions")
                .map(|map| map.contains_key(&host_surface))
                .unwrap_or(false)
            || self
                .postpositions_by_first_word
                .get(&host_surface)
                .map(|patterns| patterns.iter().any(|pattern| pattern.words.len() == 1))
                .unwrap_or(false)
            || self
                .runtime
                .literal_maps
                .get("pronouns")
                .map(|map| map.contains_key(&host_surface))
                .unwrap_or(false)
        {
            return None;
        }
        Some((marker_idx, marker_end, transform))
    }

    pub(super) fn build_postposition_context(
        &self,
        raw_ids: &[u32],
    ) -> Option<PostpositionContext> {
        if self.postpositions_by_first_word.is_empty() {
            return None;
        }

        let mut context = PostpositionContext {
            transform_for_host: vec![None; raw_ids.len()],
            host_for_raw_idx: vec![None; raw_ids.len()],
            skip_raw_idx: vec![false; raw_ids.len()],
        };
        for host_idx in 0..raw_ids.len() {
            if context.skip_raw_idx[host_idx] {
                continue;
            }
            if let Some((marker_idx, marker_end, transform)) =
                self.postposition_after(raw_ids, host_idx)
            {
                // The old implementation searched backwards from the output
                // position until the host. Record that host once per raw
                // position instead, preserving the behavior without
                // repeating the search or cloning the transform for every
                // position.
                context.transform_for_host[host_idx] = Some(transform);
                for raw_idx in host_idx..marker_idx {
                    context.host_for_raw_idx[raw_idx] = Some(host_idx);
                }
                for raw_idx in (host_idx + 1)..marker_end {
                    context.skip_raw_idx[raw_idx] = true;
                }
            }
        }
        Some(context)
    }

    fn word_surface(&self, raw_ids: &[u32], start_idx: usize, end_idx: Option<usize>) -> String {
        let mut end_idx = end_idx.unwrap_or_else(|| self.word_span_end(raw_ids, start_idx));
        if end_idx > raw_ids.len() {
            end_idx = raw_ids.len();
        }
        let decoded = self
            .decode_token_bytes(&raw_ids[start_idx..end_idx])
            .unwrap_or_else(|| self.decode_ids(&raw_ids[start_idx..end_idx]));
        decoded.trim().to_lowercase()
    }

    pub(super) fn word_span_end(&self, raw_ids: &[u32], start_idx: usize) -> usize {
        if start_idx >= raw_ids.len() {
            return start_idx;
        }
        if self.token_meta_ref(raw_ids[start_idx]).is_byte_fallback {
            let mut end = start_idx;
            while end < raw_ids.len() {
                let meta = self.token_meta_ref(raw_ids[end]);
                if meta.is_whitespace_only || meta.has_space_prefix {
                    break;
                }
                let component_end = self.byte_component_end(raw_ids, end);
                if !self.byte_component_has_word_char(raw_ids, end) {
                    break;
                }
                end = component_end;
            }
            return end.max(start_idx + 1);
        }
        let mut end = start_idx + 1;
        while end < raw_ids.len() {
            let meta = self.token_meta_ref(raw_ids[end]);
            if meta.has_space_prefix || !meta.has_word_char || meta.is_whitespace_only {
                break;
            }
            end += 1;
        }
        end
    }

    pub(super) fn previous_position_is_function_word(&self, raw_ids: &[u32], idx: usize) -> bool {
        if idx == 0 {
            return false;
        }
        let mut cursor = idx;
        while cursor > 0 && self.token_meta_ref(raw_ids[cursor - 1]).is_whitespace_only {
            cursor -= 1;
        }
        if cursor == 0 || !self.raw_position_has_word_char(raw_ids, cursor - 1) {
            return false;
        }
        let previous_meta = self.token_meta_ref(raw_ids[cursor - 1]);
        if cursor > 1
            && self.token_meta_ref(raw_ids[cursor - 2]).has_word_char
            && !previous_meta.has_space_prefix
        {
            return false;
        }
        // Metadata derived from the base BPE already contains these literal
        // matches.  Explicit metadata may omit them, so retain the decoded
        // lookup fallback for that compatibility path.
        if !self.token_meta_is_explicit && !previous_meta.is_byte_fallback {
            return previous_meta.determiner.is_some()
                || previous_meta.pronoun.is_some()
                || previous_meta.preposition.is_some()
                || previous_meta.postposition.is_some();
        }
        let surface = self.word_surface(raw_ids, cursor - 1, None);
        ["determiners", "pronouns", "prepositions", "postpositions"]
            .iter()
            .any(|family| {
                self.runtime
                    .literal_maps
                    .get(*family)
                    .map(|map| map.contains_key(&surface))
                    .unwrap_or(false)
            })
    }
}
