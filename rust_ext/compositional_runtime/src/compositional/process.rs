use super::*;

impl CompositionalTokenizer {
    fn apply_postposition(
        &self,
        context: &PostpositionContext,
        raw_idx: usize,
        modifier_rows: &mut Vec<Vec<u16>>,
    ) {
        let Some(Some(host_idx)) = context.host_for_raw_idx.get(raw_idx) else {
            return;
        };
        let Some(Some(transform)) = context.transform_for_host.get(*host_idx) else {
            return;
        };
        if let Some(group_idx) = self.group_idx(&transform.group_name) {
            if let Some(last) = modifier_rows.last_mut() {
                last[group_idx] = transform.rel_idx as u16;
            }
        }
    }

    pub(super) fn process_ids_impl(&self, raw_ids: &[u32]) -> (Vec<u32>, Vec<Vec<u16>>) {
        let mut out_ids = Vec::new();
        let mut out_mods = Vec::new();
        let space_prefix_prefix_sum: Vec<usize> = {
            let mut out = vec![0usize; raw_ids.len() + 1];
            let mut running = 0usize;
            for (idx, token_id) in raw_ids.iter().enumerate() {
                if self.token_meta_ref(*token_id).has_space_prefix {
                    running += 1;
                }
                out[idx + 1] = running;
            }
            out
        };
        let article_cap_idx = self.group_idx("article_capitalization");
        let prep_cap_idx = self.group_idx("prep_capitalization");
        let space_idx = self.group_idx("space_prefix");
        let has_base_capitalization = self.group_idx("base_capitalization").is_some();
        let pronoun_map = self
            .runtime
            .literal_maps
            .get("pronouns")
            .filter(|map| !map.is_empty());
        let suffix_group_name = self
            .group_idx("suffix_punctuation")
            .map(|_| "suffix_punctuation".to_string());

        let mut pending_groups: Vec<PendingGroup> = Vec::new();
        let mut pending_token_records: Vec<(usize, u32)> = Vec::new();
        let mut pending_leading_space = false;
        // Post-base adpositions are opt-in.  Keeping this as an Option avoids
        // allocating per-document maps/sets and makes the disabled path skip
        // all postposition matching work.
        let postposition_context = self.build_postposition_context(raw_ids);

        let literal_modifier =
            |this: &Self, start_idx: usize, token_id: u32, force_leading_space: bool| -> Vec<u16> {
                let mut modifier = this.empty_modifier();
                if let Some(idx) = space_idx {
                    if !this.token_meta_ref(token_id).is_whitespace_only
                        && (force_leading_space
                            || this.raw_expr_has_leading_space(raw_ids, start_idx))
                    {
                        modifier[idx] = 1;
                    }
                }
                modifier
            };

        let emit_literal = |this: &Self,
                            start_idx: usize,
                            token_id: u32,
                            force_leading_space: bool,
                            out_ids: &mut Vec<u32>,
                            out_mods: &mut Vec<Vec<u16>>| {
            out_ids.push(token_id);
            out_mods.push(literal_modifier(
                this,
                start_idx,
                token_id,
                force_leading_space,
            ));
        };

        let flush_pending_literal =
            |this: &Self,
             pending_leading_space_ref: &mut bool,
             pending_groups_ref: &mut Vec<PendingGroup>,
             pending_token_records_ref: &mut Vec<(usize, u32)>,
             out_ids_ref: &mut Vec<u32>,
             out_mods_ref: &mut Vec<Vec<u16>>| {
                let mut emit_leading_space = *pending_leading_space_ref;
                for (raw_idx, raw_token_id) in pending_token_records_ref.iter().copied() {
                    let is_whitespace_only = this.token_meta_ref(raw_token_id).is_whitespace_only;
                    emit_literal(
                        this,
                        raw_idx,
                        raw_token_id,
                        emit_leading_space && !is_whitespace_only,
                        out_ids_ref,
                        out_mods_ref,
                    );
                    if emit_leading_space {
                        emit_leading_space = false;
                    }
                }
                pending_groups_ref.clear();
                pending_token_records_ref.clear();
                *pending_leading_space_ref = false;
            };

        let mark_pending_detached_prefix =
            |this: &Self, start_idx: usize, pending_leading_space_ref: &mut bool| {
                if !*pending_leading_space_ref
                    && this.raw_expr_has_leading_space(raw_ids, start_idx)
                {
                    *pending_leading_space_ref = true;
                }
            };

        let pending_detached_boundary_is_representable =
            |this: &Self,
             pending_groups_ref: &[PendingGroup],
             pending_token_records_ref: &[(usize, u32)],
             host_token_id: u32| {
                let has_detached_group = pending_groups_ref.iter().any(|group| {
                    this.group_names
                        .get(group.group_idx)
                        .map(|name| {
                            matches!(
                                name.as_str(),
                                "determiners"
                                    | "pronouns"
                                    | "article_det"
                                    | "articles"
                                    | "prepositions"
                            )
                        })
                        .unwrap_or(false)
                });
                if !has_detached_group {
                    return true;
                }
                let mut count =
                    leading_ascii_spaces(&this.token_meta_ref(host_token_id).token_text);
                for (_, token_id) in pending_token_records_ref.iter().rev() {
                    let meta = this.token_meta_ref(*token_id);
                    if !meta.is_whitespace_only {
                        break;
                    }
                    if !meta.is_single_ascii_space {
                        return false;
                    }
                    count += 1;
                }
                count == 1
            };

        let normalize_pending_space =
            |modifier: &mut Vec<u16>, pending_groups_ref: &[PendingGroup], pending_space: bool| {
                if pending_groups_ref.is_empty() {
                    return;
                }
                if let Some(space_group_idx) = space_idx {
                    modifier[space_group_idx] = if pending_space {
                        1
                    } else {
                        self.default_modifier[space_group_idx]
                    };
                }
            };

        let mut idx = 0usize;
        while idx < raw_ids.len() {
            if let Some(context) = postposition_context.as_ref() {
                if context.skip_raw_idx[idx] {
                    idx += 1;
                    continue;
                }
            }
            let token_id = raw_ids[idx];
            let meta = self.token_meta_ref(token_id).clone();

            if meta.is_whitespace_only {
                if meta.token_text == " " {
                    if !pending_groups.is_empty() {
                        pending_token_records.push((idx, token_id));
                    } else if !pending_leading_space
                        && self.token_can_host_expr_space(raw_ids, idx + 1)
                    {
                        pending_leading_space = true;
                    } else {
                        emit_literal(self, idx, token_id, false, &mut out_ids, &mut out_mods);
                    }
                    idx += 1;
                    continue;
                }
                if !pending_groups.is_empty() || !pending_token_records.is_empty() {
                    flush_pending_literal(
                        self,
                        &mut pending_leading_space,
                        &mut pending_groups,
                        &mut pending_token_records,
                        &mut out_ids,
                        &mut out_mods,
                    );
                }
                emit_literal(self, idx, token_id, false, &mut out_ids, &mut out_mods);
                idx += 1;
                continue;
            }

            if meta.is_byte_fallback {
                let component_end = self.byte_component_end(raw_ids, idx);
                if let Some(component_surface) =
                    self.decode_token_bytes(&raw_ids[idx..component_end])
                {
                    let canonical_surface = component_surface.trim().to_lowercase();
                    let next_idx = component_end;
                    if let Some(transform) = self
                        .runtime
                        .literal_maps
                        .get("prefix_punctuation")
                        .and_then(|map| map.get(&canonical_surface))
                        .cloned()
                    {
                        if next_idx < raw_ids.len()
                            && self.raw_position_has_word_char(raw_ids, next_idx)
                        {
                            pending_groups.push(PendingGroup {
                                group_idx: self
                                    .group_idx(&transform.group_name)
                                    .unwrap_or(usize::MAX),
                                rel_idx: transform.rel_idx as u16,
                            });
                            if pending_groups.last().map(|group| group.group_idx)
                                == Some(usize::MAX)
                            {
                                pending_groups.pop();
                            } else {
                                for token_idx in idx..component_end {
                                    pending_token_records.push((token_idx, raw_ids[token_idx]));
                                }
                                idx = component_end;
                                continue;
                            }
                        }
                    }
                    let marker_transform = self
                        .runtime
                        .literal_maps
                        .get("determiners")
                        .and_then(|map| map.get(&canonical_surface))
                        .cloned()
                        .or_else(|| {
                            self.runtime
                                .literal_maps
                                .get("prepositions")
                                .and_then(|map| map.get(&canonical_surface))
                                .cloned()
                        })
                        .or_else(|| {
                            pronoun_map
                                .and_then(|map| map.get(&canonical_surface))
                                .cloned()
                        });
                    if let Some(transform) = marker_transform {
                        if self.can_attach_detached_modifier(
                            raw_ids,
                            idx,
                            component_end - idx,
                            !pending_groups.is_empty(),
                        ) {
                            mark_pending_detached_prefix(self, idx, &mut pending_leading_space);
                            if let Some(group_idx) = self.group_idx(&transform.group_name) {
                                pending_groups.push(PendingGroup {
                                    group_idx,
                                    rel_idx: transform.rel_idx as u16,
                                });
                                for token_idx in idx..component_end {
                                    pending_token_records.push((token_idx, raw_ids[token_idx]));
                                }
                                idx = component_end;
                                continue;
                            }
                        }
                    }
                }
                let mut first_modifier =
                    if !pending_groups.is_empty() || !pending_token_records.is_empty() {
                        self.combine_pending(&self.empty_modifier(), &pending_groups)
                    } else {
                        self.empty_modifier()
                    };
                if let Some(space_group_idx) = space_idx {
                    if pending_leading_space {
                        first_modifier[space_group_idx] = 1;
                    }
                }
                for (component_idx, token_id) in
                    raw_ids.iter().enumerate().take(component_end).skip(idx)
                {
                    out_ids.push(*token_id);
                    if component_idx == idx {
                        out_mods.push(first_modifier.clone());
                    } else {
                        out_mods.push(self.empty_modifier());
                    }
                }
                if let Some(context) = postposition_context.as_ref() {
                    self.apply_postposition(
                        context,
                        component_end.saturating_sub(1),
                        &mut out_mods,
                    );
                }
                pending_groups.clear();
                pending_token_records.clear();
                pending_leading_space = false;
                idx = component_end;
                continue;
            }

            if meta.suffix_punctuation.is_none()
                && !meta.has_word_char
                && !meta.is_whitespace_only
                && !meta.has_space_prefix
                && !out_mods.is_empty()
            {
                if let Some(suffix_map) = self.runtime.literal_maps.get("suffix_punctuation") {
                    // Match the literal against the source spelling.  The
                    // canonical surface is lower-cased and is appropriate
                    // for word transforms, but using it here would make an
                    // unsupported ``'S`` token look like the metadata's
                    // lower-case ``'s`` suffix transform.
                    let surface = meta.token_text.trim_start_matches([' ', 'Ġ', '▁']);
                    let mut best_prefix: Option<(&String, &LiteralTransform)> = None;
                    for (literal, transform) in suffix_map {
                        if literal.is_empty() || !surface.starts_with(literal) {
                            continue;
                        }
                        if best_prefix
                            .as_ref()
                            .map(|(best_literal, _)| literal.len() > best_literal.len())
                            .unwrap_or(true)
                        {
                            best_prefix = Some((literal, transform));
                        }
                    }
                    if let Some((literal, suffix_transform)) = best_prefix {
                        let remainder = &surface[literal.len()..];
                        if !remainder.is_empty() {
                            let prev_is_whitespace = idx == 0
                                || self.token_meta_ref(raw_ids[idx - 1]).is_whitespace_only;
                            let already_has_suffix = suffix_group_name
                                .as_ref()
                                .map(|name| {
                                    self.modifier_has_active_group(out_mods.last().unwrap(), name)
                                })
                                .unwrap_or(false);
                            if !prev_is_whitespace
                                && !self.previous_position_is_function_word(raw_ids, idx)
                                && !already_has_suffix
                                && self.runtime.attachment_limits.max_suffix_punctuation > 0
                            {
                                if let Some(group_idx) =
                                    self.group_idx(&suffix_transform.group_name)
                                {
                                    let remainder_ids =
                                        self.encode_segment(remainder).unwrap_or_default();
                                    if !remainder_ids.is_empty() {
                                        out_mods.last_mut().unwrap()[group_idx] =
                                            suffix_transform.rel_idx as u16;
                                        for remainder_id in remainder_ids {
                                            out_ids.push(remainder_id);
                                            out_mods.push(self.empty_modifier());
                                        }
                                        idx += 1;
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if let Some(suffix_transform) = meta.suffix_punctuation.clone() {
                if self.group_idx("suffix_punctuation").is_some() && !out_mods.is_empty() {
                    let prev_is_whitespace =
                        idx == 0 || self.token_meta_ref(raw_ids[idx - 1]).is_whitespace_only;
                    let already_has_suffix = suffix_group_name
                        .as_ref()
                        .map(|name| self.modifier_has_active_group(out_mods.last().unwrap(), name))
                        .unwrap_or(false);
                    if !prev_is_whitespace
                        && !self.previous_position_is_function_word(raw_ids, idx)
                        && !meta.has_space_prefix
                        && !already_has_suffix
                        && self.suffix_literal_matches_raw(token_id, &suffix_transform)
                        && self.runtime.attachment_limits.max_suffix_punctuation > 0
                    {
                        if let Some(group_idx) = self.group_idx(&suffix_transform.group_name) {
                            out_mods.last_mut().unwrap()[group_idx] =
                                suffix_transform.rel_idx as u16;
                            idx += 1;
                            continue;
                        }
                    }
                }
            }

            if let Some(prefix_transform) = meta.prefix_punctuation.clone() {
                let next_idx = idx + 1;
                if self.runtime.attachment_limits.max_prefix_punctuation > 0
                    && next_idx < raw_ids.len()
                    && !self.token_meta_ref(raw_ids[next_idx]).is_whitespace_only
                    && !self.token_meta_ref(raw_ids[next_idx]).has_space_prefix
                    && self.raw_position_has_word_char(raw_ids, next_idx)
                {
                    if let Some(group_idx) = self.group_idx(&prefix_transform.group_name) {
                        pending_groups.push(PendingGroup {
                            group_idx,
                            rel_idx: prefix_transform.rel_idx as u16,
                        });
                        pending_token_records.push((idx, token_id));
                        idx += 1;
                        continue;
                    }
                }
                if !pending_groups.is_empty() || !pending_token_records.is_empty() {
                    flush_pending_literal(
                        self,
                        &mut pending_leading_space,
                        &mut pending_groups,
                        &mut pending_token_records,
                        &mut out_ids,
                        &mut out_mods,
                    );
                }
            }

            if let Some(det_transform) = meta.determiner.clone() {
                if (!has_base_capitalization || meta.is_base_cap_representable)
                    && self.can_attach_detached_modifier(
                        raw_ids,
                        idx,
                        1,
                        !pending_groups.is_empty(),
                    )
                {
                    mark_pending_detached_prefix(self, idx, &mut pending_leading_space);
                    if let Some(group_idx) = self.group_idx(&det_transform.group_name) {
                        pending_groups.push(PendingGroup {
                            group_idx,
                            rel_idx: det_transform.rel_idx as u16,
                        });
                        pending_token_records.push((idx, token_id));
                        if is_capitalized_surface(&meta.token_text)
                            && (!has_base_capitalization || meta.is_base_cap_representable)
                        {
                            if let Some(cap_idx) = article_cap_idx {
                                pending_groups.push(PendingGroup {
                                    group_idx: cap_idx,
                                    rel_idx: 1,
                                });
                            }
                        }
                        idx += 1;
                        continue;
                    }
                }
            }

            if let Some(pronoun_transform) = meta.pronoun.clone() {
                if (!has_base_capitalization || meta.is_base_cap_representable)
                    && self.can_attach_detached_modifier(
                        raw_ids,
                        idx,
                        1,
                        !pending_groups.is_empty(),
                    )
                {
                    mark_pending_detached_prefix(self, idx, &mut pending_leading_space);
                    if let Some(group_idx) = self.group_idx(&pronoun_transform.group_name) {
                        pending_groups.push(PendingGroup {
                            group_idx,
                            rel_idx: pronoun_transform.rel_idx as u16,
                        });
                        pending_token_records.push((idx, token_id));
                        idx += 1;
                        continue;
                    }
                }
            }

            if let Some(prep_transform) = meta.preposition.clone() {
                if (!has_base_capitalization || meta.is_base_cap_representable)
                    && self.can_attach_detached_modifier(
                        raw_ids,
                        idx,
                        1,
                        !pending_groups.is_empty(),
                    )
                {
                    mark_pending_detached_prefix(self, idx, &mut pending_leading_space);
                    if let Some(group_idx) = self.group_idx(&prep_transform.group_name) {
                        pending_groups.push(PendingGroup {
                            group_idx,
                            rel_idx: prep_transform.rel_idx as u16,
                        });
                        pending_token_records.push((idx, token_id));
                        if is_capitalized_surface(&meta.token_text)
                            && (!has_base_capitalization || meta.is_base_cap_representable)
                        {
                            if let Some(cap_idx) = prep_cap_idx {
                                pending_groups.push(PendingGroup {
                                    group_idx: cap_idx,
                                    rel_idx: 1,
                                });
                            }
                        }
                        idx += 1;
                        continue;
                    }
                }
            }

            if !pending_groups.is_empty()
                && (meta.determiner.is_some()
                    || meta.pronoun.is_some()
                    || meta.preposition.is_some())
            {
                flush_pending_literal(
                    self,
                    &mut pending_leading_space,
                    &mut pending_groups,
                    &mut pending_token_records,
                    &mut out_ids,
                    &mut out_mods,
                );
                continue;
            }

            // These helpers scan to the end of the current word.  Calling
            // them at every BPE token repeats the same scan for each token;
            // restrict them to word starts so long unspaced spans remain
            // linear rather than quadratic.  Resolve the preceding position
            // on demand: the loop consumes byte components as a unit, so a
            // document-wide word-character cache only adds a redundant pass.
            let at_word_start = idx == 0
                || meta.has_space_prefix
                || self.token_meta_ref(raw_ids[idx - 1]).is_whitespace_only
                || !self.raw_position_has_word_char(raw_ids, idx - 1);

            if meta.has_word_char && at_word_start {
                if let Some((consumed_len, lower_surface)) = self.titlecase_lower_span(raw_ids, idx)
                {
                    if self.can_attach_detached_modifier(
                        raw_ids,
                        idx,
                        consumed_len,
                        !pending_groups.is_empty(),
                    ) {
                        if let Some(transform) = self
                            .runtime
                            .literal_maps
                            .get("determiners")
                            .and_then(|m| m.get(&lower_surface))
                            .or_else(|| {
                                self.runtime
                                    .literal_maps
                                    .get("prepositions")
                                    .and_then(|m| m.get(&lower_surface))
                            })
                            .or_else(|| pronoun_map.and_then(|map| map.get(&lower_surface)))
                        {
                            mark_pending_detached_prefix(self, idx, &mut pending_leading_space);
                            if let Some(group_idx) = self.group_idx(&transform.group_name) {
                                pending_groups.push(PendingGroup {
                                    group_idx,
                                    rel_idx: transform.rel_idx as u16,
                                });
                                if transform.group_name == "determiners"
                                    || transform.group_name == "article_det"
                                    || transform.group_name == "articles"
                                {
                                    if let Some(cap_idx) = article_cap_idx {
                                        pending_groups.push(PendingGroup {
                                            group_idx: cap_idx,
                                            rel_idx: 1,
                                        });
                                    }
                                } else if transform.group_name == "prepositions" {
                                    if let Some(cap_idx) = prep_cap_idx {
                                        pending_groups.push(PendingGroup {
                                            group_idx: cap_idx,
                                            rel_idx: 1,
                                        });
                                    }
                                }
                                for offset in 0..consumed_len {
                                    pending_token_records
                                        .push((idx + offset, raw_ids[idx + offset]));
                                }
                                idx += consumed_len;
                                continue;
                            }
                        }
                    }
                }
            }

            if !pending_detached_boundary_is_representable(
                self,
                &pending_groups,
                &pending_token_records,
                token_id,
            ) {
                flush_pending_literal(
                    self,
                    &mut pending_leading_space,
                    &mut pending_groups,
                    &mut pending_token_records,
                    &mut out_ids,
                    &mut out_mods,
                );
                continue;
            }

            let mut entry =
                self.find_longest_boundary_safe_match(raw_ids, idx, &space_prefix_prefix_sum);
            if let Some(found) = entry.clone() {
                if self.should_prefer_cap_fallback_over_match(raw_ids, idx, &found) {
                    entry = None;
                } else {
                    let mut combined_modifier = self.combine_modifier_rows(&found.modifier_rows);
                    let base_surface = self
                        .decode_ids(&found.base_ids)
                        .trim_start_matches(' ')
                        .to_lowercase();
                    let raw_surface = self.decode_ids(&raw_ids[idx..idx + found.consumed_len]);

                    if found.base_ids.len() == 1
                        && self.modifier_has_only_surface_groups(&combined_modifier)
                        && self
                            .runtime
                            .literal_maps
                            .get("determiners")
                            .map(|m| m.contains_key(&base_surface))
                            .unwrap_or(false)
                        && is_base_cap_representable_surface(&raw_surface)
                        && self.can_attach_detached_modifier(
                            raw_ids,
                            idx,
                            found.consumed_len,
                            !pending_groups.is_empty(),
                        )
                    {
                        mark_pending_detached_prefix(self, idx, &mut pending_leading_space);
                        if let Some(transform) = self
                            .runtime
                            .literal_maps
                            .get("determiners")
                            .and_then(|m| m.get(&base_surface))
                        {
                            if let Some(group_idx) = self.group_idx(&transform.group_name) {
                                pending_groups.push(PendingGroup {
                                    group_idx,
                                    rel_idx: transform.rel_idx as u16,
                                });
                                for offset in 0..found.consumed_len {
                                    pending_token_records
                                        .push((idx + offset, raw_ids[idx + offset]));
                                }
                                if is_capitalized_surface(&raw_surface)
                                    && is_base_cap_representable_surface(&raw_surface)
                                {
                                    if let Some(cap_idx) = article_cap_idx {
                                        pending_groups.push(PendingGroup {
                                            group_idx: cap_idx,
                                            rel_idx: 1,
                                        });
                                    }
                                }
                                idx += found.consumed_len;
                                continue;
                            }
                        }
                    }

                    if found.base_ids.len() == 1
                        && self.modifier_has_only_surface_groups(&combined_modifier)
                        && pronoun_map
                            .map(|m| m.contains_key(&base_surface))
                            .unwrap_or(false)
                        && is_base_cap_representable_surface(&raw_surface)
                        && self.can_attach_detached_modifier(
                            raw_ids,
                            idx,
                            found.consumed_len,
                            !pending_groups.is_empty(),
                        )
                    {
                        mark_pending_detached_prefix(self, idx, &mut pending_leading_space);
                        if let Some(transform) = pronoun_map.and_then(|map| map.get(&base_surface))
                        {
                            if let Some(group_idx) = self.group_idx(&transform.group_name) {
                                pending_groups.push(PendingGroup {
                                    group_idx,
                                    rel_idx: transform.rel_idx as u16,
                                });
                                for offset in 0..found.consumed_len {
                                    pending_token_records
                                        .push((idx + offset, raw_ids[idx + offset]));
                                }
                                idx += found.consumed_len;
                                continue;
                            }
                        }
                    }

                    if found.base_ids.len() == 1
                        && self.modifier_has_only_surface_groups(&combined_modifier)
                        && self
                            .runtime
                            .literal_maps
                            .get("prepositions")
                            .map(|m| m.contains_key(&base_surface))
                            .unwrap_or(false)
                        && is_base_cap_representable_surface(&raw_surface)
                        && self.can_attach_detached_modifier(
                            raw_ids,
                            idx,
                            found.consumed_len,
                            !pending_groups.is_empty(),
                        )
                    {
                        mark_pending_detached_prefix(self, idx, &mut pending_leading_space);
                        if let Some(transform) = self
                            .runtime
                            .literal_maps
                            .get("prepositions")
                            .and_then(|m| m.get(&base_surface))
                        {
                            if let Some(group_idx) = self.group_idx(&transform.group_name) {
                                pending_groups.push(PendingGroup {
                                    group_idx,
                                    rel_idx: transform.rel_idx as u16,
                                });
                                for offset in 0..found.consumed_len {
                                    pending_token_records
                                        .push((idx + offset, raw_ids[idx + offset]));
                                }
                                if is_capitalized_surface(&raw_surface)
                                    && is_base_cap_representable_surface(&raw_surface)
                                {
                                    if let Some(cap_idx) = prep_cap_idx {
                                        pending_groups.push(PendingGroup {
                                            group_idx: cap_idx,
                                            rel_idx: 1,
                                        });
                                    }
                                }
                                idx += found.consumed_len;
                                continue;
                            }
                        }
                    }

                    combined_modifier = self.strip_invalid_detached_modifier_groups(
                        &combined_modifier,
                        raw_ids,
                        idx,
                        found.consumed_len,
                        !pending_groups.is_empty(),
                    );
                    combined_modifier = self.apply_contextual_base_cap(
                        &combined_modifier,
                        raw_ids,
                        idx,
                        found.consumed_len,
                    );
                    combined_modifier = self.apply_contextual_space_prefix(
                        &combined_modifier,
                        raw_ids,
                        idx,
                        !pending_groups.is_empty() || pending_leading_space,
                        pending_leading_space,
                    );
                    if self.entry_has_case_mismatch(
                        &found,
                        &combined_modifier,
                        raw_ids,
                        idx,
                        found.consumed_len,
                    ) {
                        entry = None;
                    } else {
                        let combined_modifier = self.strip_nonlexical_surface_groups(
                            &combined_modifier,
                            &found.base_ids,
                            raw_ids,
                            idx,
                            found.consumed_len,
                        );
                        if !pending_groups.is_empty() || pending_leading_space {
                            if found.base_ids.len() == 1 {
                                let mut merged = found.modifier_rows[0].clone();
                                merged = self.strip_invalid_detached_modifier_groups(
                                    &merged,
                                    raw_ids,
                                    idx,
                                    found.consumed_len,
                                    !pending_groups.is_empty(),
                                );
                                merged = self.apply_contextual_base_cap(
                                    &merged,
                                    raw_ids,
                                    idx,
                                    found.consumed_len,
                                );
                                merged = self.apply_contextual_space_prefix(
                                    &merged,
                                    raw_ids,
                                    idx,
                                    !pending_groups.is_empty() || pending_leading_space,
                                    pending_leading_space,
                                );
                                merged = self.strip_nonlexical_surface_groups(
                                    &merged,
                                    &found.base_ids,
                                    raw_ids,
                                    idx,
                                    found.consumed_len,
                                );
                                merged = self.combine_pending(&merged, &pending_groups);
                                normalize_pending_space(
                                    &mut merged,
                                    &pending_groups,
                                    pending_leading_space,
                                );
                                out_ids.extend(found.base_ids.iter().copied());
                                out_mods.push(merged);
                            } else {
                                let mut combined = combined_modifier.clone();
                                if let Some(space_group_idx) = space_idx {
                                    if pending_leading_space {
                                        combined[space_group_idx] = 1;
                                    }
                                }
                                combined = self.combine_pending(&combined, &pending_groups);
                                normalize_pending_space(
                                    &mut combined,
                                    &pending_groups,
                                    pending_leading_space,
                                );
                                out_ids.extend(found.base_ids.iter().copied());
                                out_mods.extend(
                                    self.spread_multi_token_modifiers(
                                        &combined,
                                        found.base_ids.len(),
                                    ),
                                );
                            }
                            if let Some(context) = postposition_context.as_ref() {
                                self.apply_postposition(
                                    context,
                                    idx + found.consumed_len.saturating_sub(1),
                                    &mut out_mods,
                                );
                            }
                        } else {
                            out_ids.extend(found.base_ids.iter().copied());
                            if found.base_ids.len() == 1 {
                                out_mods.push(combined_modifier);
                            } else {
                                let mut normalized_rows = found.modifier_rows.clone();
                                if !normalized_rows.is_empty() {
                                    normalized_rows[0] = self.apply_contextual_base_cap(
                                        &normalized_rows[0],
                                        raw_ids,
                                        idx,
                                        found.consumed_len,
                                    );
                                    normalized_rows[0] = self.apply_contextual_space_prefix(
                                        &normalized_rows[0],
                                        raw_ids,
                                        idx,
                                        pending_leading_space,
                                        pending_leading_space,
                                    );
                                    if let Some(space_group_idx) = space_idx {
                                        for row in normalized_rows.iter_mut().skip(1) {
                                            row[space_group_idx] =
                                                self.default_modifier[space_group_idx];
                                        }
                                    }
                                }
                                out_mods.extend(normalized_rows);
                            }
                            if let Some(context) = postposition_context.as_ref() {
                                self.apply_postposition(
                                    context,
                                    idx + found.consumed_len.saturating_sub(1),
                                    &mut out_mods,
                                );
                            }
                        }
                        pending_groups.clear();
                        pending_token_records.clear();
                        pending_leading_space = false;
                        idx += found.consumed_len;
                        continue;
                    }
                }
            }

            if meta.has_word_char && at_word_start {
                if let Some((consumed_len, fallback_ids, mut fallback_mods)) = self
                    .try_lowercase_cap_fallback(
                        raw_ids,
                        idx,
                        &pending_groups,
                        pending_leading_space,
                    )
                {
                    let use_pending_space = if !pending_groups.is_empty() {
                        true
                    } else {
                        pending_leading_space
                    };
                    fallback_mods[0] = self.apply_contextual_space_prefix(
                        &fallback_mods[0],
                        raw_ids,
                        idx,
                        use_pending_space,
                        pending_leading_space,
                    );
                    normalize_pending_space(
                        &mut fallback_mods[0],
                        &pending_groups,
                        pending_leading_space,
                    );
                    out_ids.extend(fallback_ids);
                    out_mods.extend(fallback_mods);
                    if let Some(context) = postposition_context.as_ref() {
                        self.apply_postposition(
                            context,
                            idx + consumed_len.saturating_sub(1),
                            &mut out_mods,
                        );
                    }
                    pending_groups.clear();
                    pending_token_records.clear();
                    pending_leading_space = false;
                    idx += consumed_len;
                    continue;
                }
            }

            let mut base_modifier = self.empty_modifier();
            if let Some(space_group_idx) = space_idx {
                if pending_leading_space && self.raw_position_has_word_char(raw_ids, idx) {
                    base_modifier[space_group_idx] = 1;
                }
            }
            base_modifier = self.apply_contextual_base_cap(&base_modifier, raw_ids, idx, 1);
            base_modifier = self.apply_contextual_space_prefix(
                &base_modifier,
                raw_ids,
                idx,
                !pending_groups.is_empty() || pending_leading_space,
                pending_leading_space,
            );
            base_modifier = self.combine_pending(&base_modifier, &pending_groups);
            normalize_pending_space(&mut base_modifier, &pending_groups, pending_leading_space);
            out_ids.push(token_id);
            out_mods.push(base_modifier);
            if let Some(context) = postposition_context.as_ref() {
                self.apply_postposition(context, idx, &mut out_mods);
            }
            pending_groups.clear();
            pending_token_records.clear();
            pending_leading_space = false;
            idx += 1;
        }

        if !pending_groups.is_empty() || !pending_token_records.is_empty() {
            flush_pending_literal(
                self,
                &mut pending_leading_space,
                &mut pending_groups,
                &mut pending_token_records,
                &mut out_ids,
                &mut out_mods,
            );
        }
        (out_ids, out_mods)
    }
}
