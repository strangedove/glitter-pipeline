"""
Character card handling for RP Pipeline.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from rp_pipeline.data.schemas import CharacterCard


class CardDatabase:
    """
    Manages a collection of character cards for RP generation.
    Supports loading, filtering, sampling, and validation.
    """
    
    def __init__(self, path: Optional[str] = None):
        """
        Initialize card database.
        
        Args:
            path: Path to JSONL file or directory containing cards
        """
        self.path = Path(path or "restructured/pipeline/data/input/cards")
        self.cards: List[CharacterCard] = []
        self._index: Dict[str, CharacterCard] = {}
        self._load()
    
    def _load(self):
        """Load cards from JSONL files."""
        if self.path.is_file():
            self._load_file(self.path)
        elif self.path.is_dir():
            for file in self.path.glob("*.jsonl"):
                self._load_file(file)
    
    def _load_file(self, file_path: Path):
        """Load cards from a single JSONL file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            card_data = json.loads(line)
                            card = CharacterCard(**card_data)
                            self.cards.append(card)
                            # Index by name for quick lookup
                            key = f"{card.assistant_name}|{card.user_name}"
                            self._index[key] = card
                        except (json.JSONDecodeError, Exception) as e:
                            # Skip invalid lines
                            continue
        except FileNotFoundError:
            pass
    
    def __len__(self) -> int:
        return len(self.cards)
    
    def __iter__(self) -> Iterator[CharacterCard]:
        return iter(self.cards)
    
    def __getitem__(self, idx: int) -> CharacterCard:
        return self.cards[idx]
    
    def get_by_name(self, assistant_name: str, user_name: str) -> Optional[CharacterCard]:
        """Get a card by character names."""
        key = f"{assistant_name}|{user_name}"
        return self._index.get(key)
    
    def filter(
        self,
        genre: Optional[str] = None,
        tone: Optional[str] = None,
        source_model: Optional[str] = None,
        min_character_length: int = 0,
        limit: Optional[int] = None,
    ) -> List[CharacterCard]:
        """
        Filter cards by criteria.
        
        Args:
            genre: Filter by genre
            tone: Filter by tone
            source_model: Filter by source model
            min_character_length: Minimum length for character descriptions
            limit: Maximum number of results
        
        Returns:
            Filtered list of cards
        """
        result = []
        for card in self.cards:
            if genre and card.genre != genre:
                continue
            if tone and card.tone != tone:
                continue
            if source_model and card.source_model != source_model:
                continue
            if min_character_length > 0:
                if (len(card.assistant_character) < min_character_length or 
                    len(card.user_character) < min_character_length):
                    continue
            result.append(card)
            if limit and len(result) >= limit:
                break
        return result
    
    def sample(
        self,
        n: int,
        strategy: str = "random",
        genre: Optional[str] = None,
        tone: Optional[str] = None,
    ) -> List[CharacterCard]:
        """
        Sample n cards using specified strategy.
        
        Args:
            n: Number of cards to sample
            strategy: Sampling strategy ("random", "stratified", "round-robin")
            genre: Filter by genre (for stratified sampling)
            tone: Filter by tone (for stratified sampling)
        
        Returns:
            Sampled list of cards
        """
        import random
        
        if strategy == "random":
            return random.sample(self.cards, min(n, len(self.cards)))
        
        elif strategy == "stratified":
            # Stratified sampling by genre or tone
            if genre:
                return self._stratified_sample(n, "genre")
            elif tone:
                return self._stratified_sample(n, "tone")
            else:
                # Default to genre
                return self._stratified_sample(n, "genre")
        
        else:
            # Default to random
            return random.sample(self.cards, min(n, len(self.cards)))
    
    def _stratified_sample(self, n: int, attribute: str) -> List[CharacterCard]:
        """Stratified sampling by attribute."""
        import random
        from collections import defaultdict
        
        # Group by attribute
        groups = defaultdict(list)
        for card in self.cards:
            attr_value = getattr(card, attribute, None)
            if attr_value:
                groups[attr_value].append(card)
        
        # Sample proportionally from each group
        result = []
        remaining = n
        
        for group_name, group_cards in groups.items():
            if remaining <= 0:
                break
            # Proportional allocation
            count = max(1, round(n * len(group_cards) / len(self.cards)))
            count = min(count, len(group_cards), remaining)
            result.extend(random.sample(group_cards, count))
            remaining -= count
        
        # If we didn't get enough, fill with random
        if remaining > 0:
            remaining_cards = [c for c in self.cards if c not in result]
            result.extend(random.sample(remaining_cards, min(remaining, len(remaining_cards))))
        
        return result
    
    def stats(self) -> Dict[str, Any]:
        """
        Get statistics about the card collection.
        
        Returns:
            Dictionary with various statistics
        """
        from collections import Counter
        
        genres = Counter(card.genre for card in self.cards if card.genre)
        tones = Counter(card.tone for card in self.cards if card.tone)
        sources = Counter(card.source_model for card in self.cards if card.source_model)
        
        # Character length stats
        assistant_lengths = [len(card.assistant_character) for card in self.cards]
        user_lengths = [len(card.user_character) for card in self.cards]
        
        return {
            "total_cards": len(self.cards),
            "genres": dict(genres),
            "tones": dict(tones),
            "sources": dict(sources),
            "avg_assistant_length": sum(assistant_lengths) / len(assistant_lengths) if assistant_lengths else 0,
            "avg_user_length": sum(user_lengths) / len(user_lengths) if user_lengths else 0,
            "min_assistant_length": min(assistant_lengths) if assistant_lengths else 0,
            "max_assistant_length": max(assistant_lengths) if assistant_lengths else 0,
        }
    
    def save(self, path: str):
        """
        Save cards to JSONL file.
        
        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            for card in self.cards:
                f.write(card.model_dump_json() + '\n')
    
    def add(self, card: CharacterCard):
        """Add a card to the database."""
        self.cards.append(card)
        key = f"{card.assistant_name}|{card.user_name}"
        self._index[key] = card
    
    def remove(self, card: CharacterCard):
        """Remove a card from the database."""
        if card in self.cards:
            self.cards.remove(card)
        key = f"{card.assistant_name}|{card.user_name}"
        self._index.pop(key, None)


class CardFormatter:
    """Formats character cards for use in prompts."""
    
    @staticmethod
    def format_card(card: CharacterCard) -> tuple[str, str, str]:
        """
        Format a card for use in generation prompts.
        
        Returns:
            Tuple of (system_prompt_part, assistant_name, user_name)
        """
        parts = [
            f"You are {card.assistant_name} in this roleplay with {card.user_name}.",
            "",
            f"{card.assistant_name}'s profile: {card.assistant_character}",
        ]
        
        if card.assistant_appearance:
            parts.append(f"{card.assistant_name}'s appearance (keep consistent): {card.assistant_appearance}")
        
        parts.append("")
        parts.append(f"{card.user_name}'s profile: {card.user_character}")
        
        if card.user_appearance:
            parts.append(f"{card.user_name}'s appearance (keep consistent): {card.user_appearance}")
        
        parts.append("")
        parts.append(f"Scenario: {card.scenario}")
        
        return "\n".join(parts), card.assistant_name, card.user_name
    
    @staticmethod
    def format_direction(card: CharacterCard, direction: str) -> str:
        """Format a card with a specific direction."""
        card_text, _, _ = CardFormatter.format_card(card)
        return f"{card_text}\n\nDirection: {direction}"
