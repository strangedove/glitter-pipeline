"""
Analysis module for RP Pipeline.
Handles tic detection, emotion-telling analysis, and quality metrics.
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from rp_pipeline.config.settings import get_settings
from rp_pipeline.data.schemas import (
    CharacterCard,
    Scene,
    TicDetectionResult,
    Turn,
)


class TicDetector:
    """
    Detects common AI writing tics and emotion-telling patterns.
    """
    
    # Common tic patterns to detect
    TIC_PATTERNS = {
        # Emotion telling
        "emotion_telling": [
            r'\b(felt|feels|feeling)\s+',
            r'\b(knew|knows|knowing)\s+',
            r'\b(realized|realizes|realizing)\s+',
            r'\b(understood|understands)\s+',
            r'\b(something\s+shifted)\b',
            r'\b(the\s+silence\s+stretched)\b',
            r'\b(heart\s+(pounded|raced|sank|leapt))\b',
            r'\b(stomach\s+(dropped|twisted|clenched))\b',
            r'\b(breath\s+(caught|hitched|stuttered))\b',
        ],
        # Filter words / hedging
        "hedging": [
            r'\b(somehow)\b',
            r'\b(somewhat)\b',
            r'\b(quite)\b',
            r'\b(rather)\b',
            r'\b(pretty)\b',
            r'\b(a bit)\b',
            r'\b(a little)\b',
        ],
        # Lazy comparisons
        "lazy_comparisons": [
            r'\blike\s+a\s+\w+',
            r'\bthe\s+way\s+a\s+\w+',
            r'\bthe\s+way\s+he\s+\w+',
            r'\bthe\s+way\s+she\s+\w+',
            r'\bthe\s+way\s+it\s+\w+',
            r'\bthe\s+way\s+the\s+\w+',
        ],

        # Adjective-pair stacking: "<adj>, <adj> <noun>" (modern-model house style)
        "adjective_pairs": [
            r"\b(cold|clinical|quiet|slow|soft|sharp|steady|careful|deliberate|practiced|weary|grim|pale|dark|bright|warm|heavy|light|thin|thick|smooth|rough|clean|neat|precise|efficient|hollow|empty|low|distant|remote|detached|flat|dry|tight|loose|firm|gentle|harsh|crisp|muted|dense|sparse|lean|taut|slack|calm|still|silent|sudden|quick|brief|long|small|tiny|vast|immense|narrow|wide|straight|crooked|simple|plain|rich|poor|old|young|new|fresh|stale|sweet|bitter|metallic|keen|patient|methodical|mechanical|professional|military|economical|elegant|graceful|certain),\s+(cold|clinical|quiet|slow|soft|sharp|steady|careful|deliberate|practiced|weary|grim|pale|dark|bright|warm|heavy|light|thin|thick|smooth|rough|clean|neat|precise|efficient|hollow|empty|low|distant|remote|detached|flat|dry|tight|loose|firm|gentle|harsh|crisp|muted|dense|sparse|lean|taut|slack|calm|still|silent|sudden|quick|brief|long|small|tiny|vast|immense|narrow|wide|straight|crooked|simple|plain|rich|poor|old|young|new|fresh|stale|sweet|bitter|metallic|keen|patient|methodical|mechanical|professional|military|economical|elegant|graceful|certain)\s+[a-z]+\b",
        ],
        # Narrator intrusion
        "narrator_intrusion": [
            r'\b(Not\s+a\s+question\.?)\b',
            r'\b(It\s+was\s+clear\s+that)\b',
            r'\b(It\s+was\s+obvious\s+that)\b',
            r'\b(What\s+he\s+didn\'t\s+say)\b',
            r'\b(What\s+she\s+didn\'t\s+say)\b',
        ],
        # Physical cliches
        "physical_cliches": [
            r'\b(ran\s+a\s+hand\s+through\s+her\s+hair)\b',
            r'\b(ran\s+a\s+hand\s+through\s+his\s+hair)\b',
            r'\b(took\s+a\s+deep\s+breath)\b',
            r'\b(let\s+out\s+a\s+breath)\b',
            r'\b(held\s+her\s+breath)\b',
            r'\b(held\s+his\s+breath)\b',
            r'\b(bit\s+her\s+lip)\b',
            r'\b(bit\s+his\s+lip)\b',
            r'\b(clenched\s+her\s+jaw)\b',
            r'\b(clenched\s+his\s+jaw)\b',
        ],
        # Dialogue tags
        "dialogue_tags": [
            r'\b(said,\s+her\s+voice)\b',
            r'\b(said,\s+his\s+voice)\b',
            r'\b(said,\s+voice\s+\w+)\b',
            r'\b(asked,\s+her\s+voice)\b',
            r'\b(asked,\s+his\s+voice)\b',
        ],
        # Pronoun repetition (sentence openers)
        "pronoun_repetition": [
            r'^\s*(He\s)',
            r'^\s*(She\s)',
            r'^\s*(They\s)',
            r'^\s*(The\s)',
            r'^\s*(I\s)',
        ],
        # Pivot to warmth
        "pivot_to_warmth": [
            r'\b(softened)\b',
            r'\b(warmed)\b',
            r'\b(melted)\b',
            r'\b(thawed)\b',
        ],
    }
    
    # Emotion words that indicate telling vs showing
    EMOTION_WORDS = [
        'angry', 'happy', 'sad', 'afraid', 'scared', 'nervous', 'anxious',
        'excited', 'surprised', 'confused', 'relieved', 'disappointed',
        'frustrated', 'amused', 'embarrassed', 'ashamed', 'proud',
        'jealous', 'lonely', 'hopeful', 'determined', 'resigned',
    ]
    
    def __init__(self):
        """Initialize the tic detector."""
        self.settings = get_settings()
        # Compile patterns for efficiency
        self._compiled_patterns = {
            category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for category, patterns in self.TIC_PATTERNS.items()
        }
    
    def detect_tics(
        self,
        text: str,
        category: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """
        Detect tics in a text string.
        
        Args:
            text: Text to analyze
            category: Optional category to check (checks all if None)
        
        Returns:
            Dictionary mapping tic type to list of matched strings
        """
        if category:
            patterns = {category: self._compiled_patterns.get(category, [])}
        else:
            patterns = self._compiled_patterns
        
        results = {}
        for tic_type, compiled_patterns in patterns.items():
            matches = []
            for pattern in compiled_patterns:
                for match in pattern.finditer(text):
                    matched_text = match.group()
                    # Only add unique matches
                    if matched_text not in matches:
                        matches.append(matched_text)
            if matches:
                results[tic_type] = matches
        
        return results
    
    def detect_emotion_telling(self, text: str) -> List[str]:
        """
        Detect emotion-telling instances (narrator explaining emotions).
        
        Args:
            text: Text to analyze
        
        Returns:
            List of detected emotion-telling phrases
        """
        emotion_patterns = [
            # "she felt happy" style
            rf'\b({"|".join(self.EMOTION_WORDS)})\b',
            # "felt angry"
            r'\b(felt|feels|feeling)\s+\w+',
            # "was angry"
            r'\b(was|were|is|are)\s+(\w+\s+)?({"|".join(self.EMOTION_WORDS)})\b',
        ]
        
        results = []
        for pattern in emotion_patterns:
            for match in re.compile(pattern, re.IGNORECASE).finditer(text):
                results.append(match.group())
        
        return results
    
    def check_pronoun_repetition(
        self,
        text: str,
        max_consecutive: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Check for repeated sentence openers (pronoun repetition).
        
        Args:
            text: Text to analyze
            max_consecutive: Maximum allowed consecutive same openers
        
        Returns:
            List of issues found
        """
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        issues = []
        current_streak: List[str] = []
        
        for sentence in sentences:
            # Get first word
            first_word = sentence.split()[0] if sentence.split() else ""
            first_word = re.sub(r'[^\w]', '', first_word).lower()
            
            if not first_word:
                continue
            
            if current_streak and current_streak[-1][0] == first_word:
                current_streak.append(first_word)
            else:
                if len(current_streak) >= max_consecutive:
                    issues.append({
                        "opener": current_streak[0],
                        "count": len(current_streak),
                        "example": " ".join([s for s in current_streak])
                    })
                current_streak = [first_word]
        
        # Check final streak
        if len(current_streak) >= max_consecutive:
            issues.append({
                "opener": current_streak[0],
                "count": len(current_streak),
                "example": " ".join([s for s in current_streak])
            })
        
        return issues
    
    def analyze_scene(self, scene: Scene) -> TicDetectionResult:
        """
        Analyze a scene for tics and quality issues.
        
        Args:
            scene: Scene to analyze
        
        Returns:
            TicDetectionResult with findings
        """
        # Focus on assistant turns only
        assistant_text = " ".join(
            t.content for t in scene.turns if t.role == "ASSISTANT"
        )
        
        # Detect all tics
        all_tics = self.detect_tics(assistant_text)
        
        # Detect emotion telling
        emotion_tells = self.detect_emotion_telling(assistant_text)
        
        # Check pronoun repetition
        pronoun_issues = self.check_pronoun_repetition(assistant_text)
        if pronoun_issues:
            all_tics["pronoun_repetition"] = [
                f"{issue['opener']} x{issue['count']}"
                for issue in pronoun_issues
            ]
        
        # Count total tics
        total_tic_count = sum(len(matches) for matches in all_tics.values())
        
        # Calculate tic rate (per 1000 words)
        word_count = scene.total_word_count
        tic_rate = (total_tic_count / word_count * 1000) if word_count > 0 else 0.0
        
        # Determine if cleanup is needed
        quality_config = self.settings.quality
        needs_cleanup = tic_rate > quality_config.get("tic_rate_threshold", 5.0)
        
        return TicDetectionResult(
            scene_id=scene.card_id or str(id(scene)),
            tics={k: len(v) for k, v in all_tics.items()},
            emotion_tells=emotion_tells,
            total_tic_count=total_tic_count,
            tic_rate=round(tic_rate, 2),
            needs_cleanup=needs_cleanup,
        )
    
    def analyze_batch(
        self,
        scenes: List[Scene],
    ) -> List[TicDetectionResult]:
        """
        Analyze multiple scenes.
        
        Args:
            scenes: List of scenes to analyze
        
        Returns:
            List of analysis results
        """
        return [self.analyze_scene(scene) for scene in scenes]


class QualityAnalyzer:
    """
    Analyzes scene quality beyond tic detection.
    """
    
    def __init__(self):
        """Initialize quality analyzer."""
        self.settings = get_settings()
    
    def check_turn_variety(self, scene: Scene) -> Tuple[bool, List[str]]:
        """
        Check for turn length variety.
        
        Args:
            scene: Scene to check
        
        Returns:
            Tuple of (passes, list of issues)
        """
        issues = []
        assistant_turns = [t for t in scene.turns if t.role == "ASSISTANT"]
        
        if len(assistant_turns) < 2:
            return True, issues
        
        word_counts = [t.word_count for t in assistant_turns]
        
        # Check for uniformity (all turns roughly same length)
        avg_count = sum(word_counts) / len(word_counts)
        variance = sum((c - avg_count) ** 2 for c in word_counts) / len(word_counts)
        std_dev = variance ** 0.5
        
        # If standard deviation is less than 20% of average, it's too uniform
        if std_dev < avg_count * 0.2:
            issues.append(
                f"Turn length uniformity: all assistant turns ~{avg_count:.0f} words "
                f"(std dev: {std_dev:.0f})"
            )
        
        return len(issues) == 0, issues
    
    def check_scene_advancement(self, scene: Scene) -> Tuple[bool, List[str]]:
        """
        Check if the scene genuinely advances.
        
        Args:
            scene: Scene to check
        
        Returns:
            Tuple of (passes, list of issues)
        """
        issues = []
        
        if len(scene.turns) < 2:
            return True, issues
        
        first_turn = scene.turns[0].content.lower()
        last_turn = scene.turns[-1].content.lower()
        
        # Simple heuristic: check if characters are in same state
        # This would be enhanced with semantic analysis
        if first_turn == last_turn:
            issues.append("Scene starts and ends identically - no advancement")
        
        return len(issues) == 0, issues
    
    def check_responsiveness(
        self,
        scene: Scene,
    ) -> Tuple[bool, List[str]]:
        """
        Check if turns respond to each other.
        
        Args:
            scene: Scene to check
        
        Returns:
            Tuple of (passes, list of issues)
        """
        issues = []
        
        for i in range(1, len(scene.turns)):
            prev_turn = scene.turns[i-1].content.lower()
            curr_turn = scene.turns[i].content.lower()
            
            # Check if current turn mentions anything from previous
            # Very simple check - would use embeddings in production
            prev_words = set(prev_turn.split())
            curr_words = set(curr_turn.split())
            
            overlap = prev_words & curr_words
            # Remove common words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'of'}
            overlap = overlap - stop_words
            
            if len(overlap) < 2 and len(prev_turn.split()) > 10:
                issues.append(
                    f"Turn {scene.turns[i].turn_number} may not respond to "
                    f"Turn {scene.turns[i-1].turn_number}"
                )
        
        return len(issues) == 0, issues
    
    def analyze_scene_quality(self, scene: Scene) -> Dict[str, Any]:
        """
        Perform comprehensive quality analysis.
        
        Args:
            scene: Scene to analyze
        
        Returns:
            Dictionary with quality metrics
        """
        result = {
            "turn_variety": {"passes": True, "issues": []},
            "scene_advancement": {"passes": True, "issues": []},
            "responsiveness": {"passes": True, "issues": []},
        }
        
        result["turn_variety"]["passes"], result["turn_variety"]["issues"] = \
            self.check_turn_variety(scene)
        
        result["scene_advancement"]["passes"], result["scene_advancement"]["issues"] = \
            self.check_scene_advancement(scene)
        
        result["responsiveness"]["passes"], result["responsiveness"]["issues"] = \
            self.check_responsiveness(scene)
        
        all_pass = all(
            check["passes"] for check in result.values()
        )
        result["overall_pass"] = all_pass
        
        return result


class SceneAnalyzer:
    """
    Combined analyzer for scenes.
    """
    
    def __init__(self):
        """Initialize scene analyzer."""
        self.tic_detector = TicDetector()
        self.quality_analyzer = QualityAnalyzer()
    
    def analyze(
        self,
        scene: Scene,
    ) -> Tuple[TicDetectionResult, Dict[str, Any]]:
        """
        Perform full analysis on a scene.
        
        Args:
            scene: Scene to analyze
        
        Returns:
            Tuple of (tic_detection_result, quality_analysis)
        """
        tic_result = self.tic_detector.analyze_scene(scene)
        quality_result = self.quality_analyzer.analyze_scene_quality(scene)
        
        return tic_result, quality_result
