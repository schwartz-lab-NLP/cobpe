"""Modifier container types for dual-stream decomposition."""

from typing import Dict, List, Optional, Tuple

from .constants import *  # type: ignore  # noqa: F401,F403

class UnifiedModifierArray:
    """Manages the 2D modifier array for dual-stream tokenization.

    The modifier array has shape (seq_len, num_groups) where each position
    contains a group-relative index for that transformation group.

    Example:
        For a token with article "the" and past tense inflection:
        modifier_array[i] = [0, 0, 2, 0, 1, 0, 0, 0, 0, 0]
        # space_prefix=0 (no space), cap=0 (no cap), inflection=2 (PAST),
        # derivation=0, articles=1 (THE), etc.
    """

    def __init__(self, groups: Optional[List[str]] = None,
                 types_loss_indices_map: Optional[Dict[str, Tuple[int, int]]] = None):
        """Initialize the unified modifier array manager.

        Args:
            groups: List of transformation group names to use.
                   Defaults to UNIFIED_TRANSFORM_GROUPS.
            types_loss_indices_map: Mapping from group names to (start_idx, end_idx)
                                   in the global transformation space.
        """
        self.groups = groups or UNIFIED_TRANSFORM_GROUPS.copy()
        self.num_groups = len(self.groups)
        self.group_to_idx = {name: i for i, name in enumerate(self.groups)}

        # Global transformation indices mapping
        self.types_loss_indices_map = types_loss_indices_map or {}

        # Group sizes (number of possible values per group)
        self.group_sizes = {}
        for group in self.groups:
            if group in self.types_loss_indices_map:
                start, end = self.types_loss_indices_map[group]
                self.group_sizes[group] = end - start
            else:
                self.group_sizes[group] = 1  # Unknown group, assume single value

    def create_empty_modifier(self) -> List[int]:
        """Create an empty modifier tuple (all zeros = no modifications)."""
        return [0] * self.num_groups

    def max_group_relative_index(self) -> int:
        """Maximum group-relative value across all active transformation groups."""
        max_idx = 0
        for group in self.groups:
            group_size = int(self.group_sizes.get(group, 1))
            if group_size > 0:
                max_idx = max(max_idx, group_size - 1)
        return max_idx

    def recommended_modifier_dtype_name(self) -> str:
        """Smallest unsigned dtype name that can represent all modifier values."""
        max_idx = self.max_group_relative_index()
        if max_idx <= 0xFF:
            return "uint8"
        if max_idx <= 0xFFFF:
            return "uint16"
        if max_idx <= 0xFFFFFFFF:
            return "uint32"
        return "uint64"

    def set_group_value(self, modifier: List[int], group_name: str, value: int) -> List[int]:
        """Set the value for a specific group in a modifier tuple.

        Args:
            modifier: The modifier tuple to modify (list of group-relative indices).
            group_name: Name of the group to set.
            value: Group-relative index for that group.

        Returns:
            Modified modifier tuple.
        """
        if group_name in self.group_to_idx:
            modifier[self.group_to_idx[group_name]] = value
        return modifier

    def get_group_value(self, modifier: List[int], group_name: str) -> int:
        """Get the value for a specific group from a modifier tuple."""
        if group_name in self.group_to_idx:
            return modifier[self.group_to_idx[group_name]]
        return 0

    def global_to_group_relative(self, group_name: str, global_idx: int) -> int:
        """Convert a global transformation index to a group-relative index.

        Args:
            group_name: Name of the transformation group.
            global_idx: Index in the global transformation space.

        Returns:
            Group-relative index (0, 1, 2, ... within the group).
        """
        if group_name not in self.types_loss_indices_map:
            return 0
        start, _ = self.types_loss_indices_map[group_name]
        return global_idx - start

    def group_relative_to_global(self, group_name: str, relative_idx: int) -> int:
        """Convert a group-relative index to a global transformation index.

        Args:
            group_name: Name of the transformation group.
            relative_idx: Index within the group (0, 1, 2, ...).

        Returns:
            Global transformation index.
        """
        if group_name not in self.types_loss_indices_map:
            return 0
        start, _ = self.types_loss_indices_map[group_name]
        return start + relative_idx

    def to_one_hot(self, modifier: List[int], total_transform_dim: int) -> List[float]:
        """Convert group-relative modifier to one-hot representation.

        Args:
            modifier: List of group-relative indices.
            total_transform_dim: Total size of the one-hot vector.

        Returns:
            One-hot encoded transformation vector.
        """
        one_hot = [0.0] * total_transform_dim
        for group_idx, group_name in enumerate(self.groups):
            if group_name in self.types_loss_indices_map:
                global_idx = self.group_relative_to_global(group_name, modifier[group_idx])
                if global_idx < total_transform_dim:
                    one_hot[global_idx] = 1.0
        return one_hot

    def from_one_hot(self, one_hot: List[float]) -> List[int]:
        """Convert one-hot representation back to group-relative modifier.

        Args:
            one_hot: One-hot encoded transformation vector.

        Returns:
            List of group-relative indices.
        """
        modifier = self.create_empty_modifier()
        for group_idx, group_name in enumerate(self.groups):
            if group_name in self.types_loss_indices_map:
                start, end = self.types_loss_indices_map[group_name]
                # Find which index in this group is set
                for i in range(start, end):
                    if i < len(one_hot) and one_hot[i] > 0.5:
                        modifier[group_idx] = i - start
                        break
        return modifier

    def to_dict(self) -> dict:
        """Serialize to dictionary for storage."""
        return {
            'groups': self.groups,
            'types_loss_indices_map': self.types_loss_indices_map,
            'group_sizes': self.group_sizes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'UnifiedModifierArray':
        """Deserialize from dictionary."""
        instance = cls(
            groups=data.get('groups', UNIFIED_TRANSFORM_GROUPS),
            types_loss_indices_map=data.get('types_loss_indices_map', {})
        )
        if 'group_sizes' in data:
            instance.group_sizes = data['group_sizes']
        return instance

    def __repr__(self):
        return f"UnifiedModifierArray(groups={self.groups}, sizes={self.group_sizes})"


__all__ = [
    'UnifiedModifierArray',
]
