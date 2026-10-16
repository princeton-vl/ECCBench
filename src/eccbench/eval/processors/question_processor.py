"""Build question prompts, ask the model, and score its answer."""

import gc
import os
import re
import string
from typing import Any, Dict, Optional

import torch

from eccbench.data.manifest import QuestionEntry

# Calibration arms, selected with CALIB_ARM: (reward, penalty, threshold %).
# Each payoff implies the confidence threshold penalty / (reward + penalty):
# 1/4 for arm A (below chance) and 4/5 for arm B (above chance).
# Without CALIB_ARM the prompt has no scoring sentence.
_CALIB_ARMS = {
    'A': (3, 1, 25),
    'B': (1, 4, 80),
}


def _rag_preamble() -> str:
    """Disclosure prepended to every prompt when RAG_PREAMBLE=1, else ''.

    It is added to the prompt, not the question text, because the natural
    video template is chosen by how the question text starts.
    """
    if os.environ.get('RAG_PREAMBLE') != '1':
        return ''
    return (
        "NOTE: A retrieval system has pre-selected only the segments of the "
        "original video most relevant to the question below. You are NOT "
        "seeing the complete video. Distinct segments are separated by a "
        "bright magenta frame labeled SEGMENT BREAK. "
    )


def _calibration_payoff_suffix(abstain_token: str) -> str:
    """Scoring sentence for the CALIB_ARM arm; '' when CALIB_ARM is unset."""
    arm = os.environ.get('CALIB_ARM', '').strip().upper()
    if not arm:
        return ''
    if arm not in _CALIB_ARMS:
        raise ValueError(
            f"CALIB_ARM={arm!r} is not one of {sorted(_CALIB_ARMS)}"
        )
    reward, penalty, threshold = _CALIB_ARMS[arm]

    def pt(n: int) -> str:
        return f"{n} point" if n == 1 else f"{n} points"

    return (
        f" Scoring: a correct answer earns +{pt(reward)}, a wrong answer "
        f"costs -{pt(penalty)}, and {abstain_token} earns 0 points. "
        f"Answer only if you are more than {threshold}% confident; "
        f"otherwise answer {abstain_token}."
    )


class QuestionProcessor:
    """Builds prompts, asks the model and scores the answer.

    Handles native binary questions (one candidate, yes/no/unsure) and the
    older multi-choice format.
    """

    def __init__(
        self,
        verbose: bool = False,
        binary_questions: bool = False,
        sequence_mode: bool = False,
    ) -> None:
        self.verbose = verbose
        self.binary_questions = binary_questions
        self.sequence_mode = sequence_mode

    def extract_answer(self, response: str) -> str:
        """Return the last answer in braces, answer tags or box tokens; "2" if none."""
        matches = re.findall(r'\{([^}]+)\}', response)
        if matches:
            raw_answer = matches[-1].strip()
            normalised = self._normalise_extracted_answer(raw_answer)
            # MiMo-VL sometimes writes \boxed{uncertain}; count it as an abstention.
            if not self.binary_questions and "\\boxed{" in response:
                cleaned = normalised.lower().lstrip('{').rstrip('}').strip()
                if cleaned in {'uncertain', 'unsure', 'unknown', 'idk', "don't know", 'dont know'}:
                    return "2"
            return normalised

        answer_tag_matches = re.findall(
            r"<answer>(.*?)</answer>",
            response,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if answer_tag_matches:
            raw_answer = answer_tag_matches[-1].strip()
            return self._normalise_extracted_answer(raw_answer)

        box_matches = re.findall(
            r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>",
            response,
            flags=re.DOTALL,
        )
        if box_matches:
            raw_answer = box_matches[-1].strip()
            return self._normalise_extracted_answer(raw_answer)

        # A bare digit, as Phi-4-MM sometimes answers in video mode.
        stripped = response.strip()
        if stripped in ['0', '1', '2']:
            return stripped

        # A bare digit after a <think> block.
        think_removed = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        think_stripped = think_removed.strip()
        if think_stripped and re.match(r'^\s*[0-2]\s*$', think_stripped):
            digit_match = re.search(r'[0-2]', think_stripped)
            if digit_match:
                return digit_match.group()

        # A standalone final digit, as in "The answer is 1." but not "answer2".
        stripped = response.strip().rstrip(string.punctuation + string.whitespace)
        if stripped:
            last_char = stripped[-1]
            if last_char in ['0', '1', '2']:
                if len(stripped) == 1:
                    return last_char
                prev_char = stripped[-2]
                if prev_char in ' .,!?;:\n\t()-':
                    return last_char

        return "2"

    def _normalise_extracted_answer(self, raw_answer: str) -> str:
        if self.binary_questions:
            return raw_answer.strip(string.punctuation + " ")

        numeric_match = re.findall(r"-?\d+", raw_answer)
        if numeric_match:
            return numeric_match[-1]

        return raw_answer.strip(string.punctuation + " ")

    def normalize_native_binary_answer(self, raw_answer: str) -> str:
        """Map an extracted answer to '0' (yes), '1' (no) or '2' (unsure or unrecognised)."""
        cleaned = raw_answer.strip(string.punctuation + " ").lower()

        if cleaned == '0':
            return '0'
        if cleaned == '1':
            return '1'
        if cleaned == '2':
            return '2'

        yes_variants = {'yes', 'y', 'true', 'correct', 'appeared', 'present'}
        no_variants = {'no', 'n', 'false', 'incorrect', 'not', 'absent', 'didnt', "didn't"}
        uncertain_variants = {'uncertain', 'idk', 'unsure', 'unknown', "don't know", 'dont know', 'maybe'}

        if cleaned in yes_variants:
            return '0'
        if cleaned in no_variants:
            return '1'
        if cleaned in uncertain_variants:
            return '2'

        if cleaned.startswith('yes'):
            return '0'
        if cleaned.startswith('no'):
            return '1'
        if 'uncertain' in cleaned or 'unsure' in cleaned:
            return '2'
        return '2'

    def build_question_text(self, question: QuestionEntry, num_options: int) -> str:
        if self.sequence_mode:
            return self._build_sequence_question_text(question, num_options)
        base_text = self._build_video_question_text(num_options)
        return self.adjust_text_for_context(base_text)

    def _build_video_question_text(self, num_options: int) -> str:
        return (
            "In the main video, letters moved down three conveyer belts. "
            "A letter that appeared on a specific conveyer belt ALWAYS stayed on the same conveyer belt and only moved down, not left or right. "
            "The letters appeared sequentially, but multiple may have been on screen at the same time. "
            "The letters continued moving down the conveyer belt until they left the screen. "
            "Examine the main video and the option video closely. "
            "Did this option video show events that exactly appeared in the main video, down to the order they appeared and which conveyer belt each letter was on? "
            "Write {0} if yes, {1} if no, or {2} if unsure. Please put your answer in curly braces, e.g, {2}."
        )

    def _convert_prompt_for_sequence_mode(self, prompt: str) -> str:
        replacements = [
            ("main video", "main sequence"),
            ("Main video", "Main sequence"),
            ("main videos", "main sequences"),
            ("Main videos", "Main sequences"),
            ("option video", "option sequence"),
            ("Option video", "Option sequence"),
            ("option videos", "option sequences"),
            ("Option videos", "Option sequences"),
            ("videos", "sequences"),
            ("Videos", "Sequences"),
            ("video", "sequence"),
            ("Video", "Sequence"),
            ("frames", "tokens"),
            ("Frames", "Tokens"),
            ("clip", "sequence"),
            ("Clip", "Sequence"),
            ("footage", "sequence context"),
            ("Footage", "Sequence context"),
        ]
        updated = prompt
        for target, replacement in replacements:
            updated = updated.replace(target, replacement)
        return updated

    def adjust_text_for_context(self, text: str) -> str:
        if self.sequence_mode:
            return self._convert_prompt_for_sequence_mode(text)
        return text

    def process_single_question(
        self,
        model: Any,
        question: QuestionEntry,
        video_index: int,
        max_tokens: int = 700,
        max_frames: int = 256,
        current_video_time: Optional[float] = 0.0,
        binary_selected_option_index: Optional[int] = None,
        binary_correct_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ask one multi-choice question and score it."""
        num_options = 2 if self.binary_questions else len(question.options)
        question_text = self.build_question_text(question, num_options)

        if self.verbose:
            self._print_question_details(question, video_index, question_text)

        effective_time = current_video_time
        if effective_time is None:
            effective_time = float(question.question_time)

        response = model.ask_question(
            question_text,
            current_video_time=effective_time,
            max_tokens=max_tokens,
            max_frames_in_video=max_frames,
        )

        video_truncated = None
        if hasattr(model, 'was_video_truncated'):
            video_truncated = model.was_video_truncated()

        if self.verbose:
            self._print_response(response)

        predicted_answer = self.extract_answer(response).strip().lower()

        if self.binary_questions:
            if binary_correct_answer is None:
                raise ValueError("Binary mode requires a provided correct answer indicator.")
            correct_answer = binary_correct_answer.strip().lower()
            if correct_answer not in {"yes", "no"}:
                raise ValueError("Binary mode requires the correct answer to be either 'yes' or 'no'.")
            is_correct = predicted_answer == correct_answer
            is_dont_know = False
        else:
            correct_answer = str(question.correct_answer_index)
            is_correct = predicted_answer == correct_answer
            is_dont_know = predicted_answer == str(question.dont_know_index)

        question_identifier = getattr(question, 'question_order', question.question_id)
        if self.binary_questions:
            selected_binary_option = binary_selected_option_index
            if selected_binary_option is not None and 0 <= selected_binary_option < len(question.options):
                option_name = os.path.basename(question.options[selected_binary_option].clip_path)
                option_display = f"Option {selected_binary_option} ({option_name})"
            else:
                option_display = "Option ?"

            predicted_binary_answer = (predicted_answer or '').lower()
            print(
                f"Q{question_identifier}: {option_display} -> "
                f"Predicted: {predicted_binary_answer.upper() or 'UNKNOWN'}, "
                f"Correct: {correct_answer.upper() or 'UNKNOWN'} - "
                f"{'CORRECT' if is_correct else 'INCORRECT'}"
            )
        elif is_dont_know:
            print(
                f"Q{question_identifier}: Predicted: {predicted_answer} (Don't Know), "
                f"Correct: {correct_answer} - IDK"
            )
        else:
            print(
                f"Q{question_identifier}: Predicted: {predicted_answer}, "
                f"Correct: {correct_answer} - {'CORRECT' if is_correct else 'INCORRECT'}"
            )

        has_answer = bool(predicted_answer)

        if hasattr(model, "record_question_outcome"):
            model.record_question_outcome(
                current_video_time=effective_time,
                is_correct=is_correct,
                is_dont_know=is_dont_know,
                is_answered=has_answer,
            )

        if self.verbose:
            if is_dont_know:
                print("Result: DON'T KNOW")
            else:
                print(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")
            print(f"{'='*60}")

        _free_memory()

        result = {
            'video_index': video_index,
            'question_id': question.question_id,
            'question_order': getattr(question, 'question_order', None),
            'predicted': predicted_answer,
            'correct': correct_answer,
            'is_correct': is_correct,
            'is_dont_know': is_dont_know,
            'num_options': num_options,
            'response': response,
            'saw_all_frames': None if video_truncated is None else not video_truncated,
            'max_tokens': max_tokens,
            'max_frames': max_frames,
        }

        if self.binary_questions:
            result['binary_selected_option_index'] = binary_selected_option_index
            result['binary_correct_answer'] = correct_answer

        return result

    def _print_question_details(self, question: QuestionEntry, video_index: int, question_text: str) -> None:
        print(f"\n{'='*60}")
        print("QUESTION DETAILS")
        print(f"{'='*60}")
        question_number = getattr(question, 'question_order', question.question_id)
        print(f"Video Index: {video_index}")
        print(f"Question #: {question_number} (ID: {question.question_id})")

        if self.binary_questions:
            print("Mode: Binary yes/no (single option clip)")
        else:
            if self.sequence_mode and question.sequence_prefixes:
                print("Sequence prefixes:")
                for key in sorted(question.sequence_prefixes.keys()):
                    tokens = question.sequence_prefixes.get(key) or []
                    normalized = ", ".join(str(token) for token in tokens)
                    print(f"  {key}: {normalized or '(empty)'}")
            print("Options:")
            for i, option in enumerate(question.options):
                if self.sequence_mode:
                    sequence_display = self._describe_option_sequence(option)
                    print(f"  {i}: {sequence_display}")
                else:
                    print(f"  {i}: {os.path.basename(option.clip_path)}")
            print(f"Correct Answer: {question.correct_answer_index}")
            print(f"Don't Know Index: {question.dont_know_index}")

        print("\nFull Question Text:")
        print(question_text)
        print(f"{'='*60}")

    def _print_response(self, response: str) -> None:
        print(f"\n{'='*60}")
        print("MODEL RESPONSE")
        print(f"{'='*60}")
        print(response)
        print(f"{'='*60}")

    def _describe_option_sequence(self, option: Any) -> str:
        tokens = []
        if getattr(option, 'token_sequence', None):
            tokens = [str(token) for token in option.token_sequence]
        elif isinstance(option.metadata, dict):
            seq = option.metadata.get('sequence')
            if isinstance(seq, list):
                tokens = [str(token) for token in seq]
            elif isinstance(option.metadata.get('sequences'), dict):
                for value in option.metadata['sequences'].values():
                    if isinstance(value, list) and value:
                        tokens = [str(token) for token in value]
                        break
        return ", ".join(tokens) if tokens else "(no sequence provided)"

    def _build_sequence_question_text(self, question: QuestionEntry, num_options: int) -> str:
        return (
            "In the main sequence, tokens appeared in a specific order. "
            "Examine the main sequence and the option sequence closely. "
            "Did this option sequence appear somewhere in the main sequence as a contiguous subsequence, in the exact same order? "
            "For example, if the main sequence was [1,2,4,3,4,4,5] and the option sequence was [3,4,5], the answer would be {1} because 3,4,5 never appears consecutively in the main sequence. "
            "However, if the option sequence was [3,4,4], the answer would be {0} because 3,4,4 does appear consecutively in the main sequence. "
            "Write {0} if yes, {1} if no, or {2} if unsure. Please put your answer in curly braces, e.g, {2}."
        )

    @staticmethod
    def format_primary_prefix(question: QuestionEntry) -> str:
        """The first prefix sequence of a continuation question, as text."""
        prefixes = question.sequence_prefixes or {}
        if not prefixes:
            return ""
        first_key = sorted(prefixes.keys())[0]
        tokens = prefixes.get(first_key)
        if not isinstance(tokens, list):
            return ""
        return ", ".join(str(token) for token in tokens)

    def build_native_binary_question_text(
        self,
        question: QuestionEntry,
        include_uncertain: bool = True,
    ) -> str:
        """Prompt for a native binary question, with the RAG and calibration additions."""
        mode = (question.question_mode or "").strip().lower()
        uncertain_text = " or {uncertain} if you are unsure" if include_uncertain else ""

        if self.sequence_mode:
            text = self._build_native_binary_sequence_text(question, mode, uncertain_text)
        else:
            text = self._build_native_binary_video_text(question, mode, uncertain_text)
        if not include_uncertain:
            return _rag_preamble() + text
        # The natural template offers '{uncertain}'; the synthetic ones offer '{2}'.
        abstain_token = '{uncertain}' if '{uncertain}' in text else '{2}'
        return _rag_preamble() + text + _calibration_payoff_suffix(abstain_token)

    def _build_native_binary_video_text(
        self,
        question: QuestionEntry,
        mode: str,
        uncertain_text: str,
    ) -> str:
        """Natural questions (no candidate clip or sequence) use their own text;
        synthetic questions use the conveyor-belt prompt."""
        candidate = question.candidate
        is_natural = (
            candidate is None
            or (not candidate.clip_path and not candidate.sequence)
        )
        use_short_prompt_for_synth = os.environ.get('USE_SHORT_PROMPT_FOR_SYNTH') == '1'
        if (is_natural or use_short_prompt_for_synth) and mode == 'exists' and question.prompt and (
            question.prompt.startswith('Did ') or
            question.prompt.startswith('Was ') or
            question.prompt.startswith('Is ')
        ):
            return (
                f"Watch this video closely, paying attention to the entire video. "
                f"{question.prompt}\n\n"
                f"Write {{0}} if yes, {{1}} if no{uncertain_text}. "
                f"Please put your answer in curly braces, e.g, {{0}}."
            )

        base_prompt = (
            "In the main video and option video, letters moved down three conveyer belts. "
            "A letter that appeared in any of the videos on a specific conveyer belt ALWAYS stayed on the same conveyer belt and only moved down, not left or right. "
            "The letters appeared sequentially. Only one was on the screen at a time. "
            "The letters continued moving down the conveyer belt until they left the screen. "
            "Examine the main video and the option video closely. "
            "Did this option video show events that exactly appeared SOMEWHERE in the main video, down to the order they appeared and which conveyer belt each letter was on? "
            "Write {0} if yes, {1} if no, or {2} if unsure. Please put your answer in curly braces, e.g, {2}."
        )

        return base_prompt

    def _build_native_binary_sequence_text(
        self,
        question: QuestionEntry,
        mode: str,
        uncertain_text: str,
    ) -> str:
        return (
            "You are given a main sequence of tokens and a candidate sequence. "
            "Your task: Determine if the candidate sequence appears as a contiguous subsequence within the main sequence, in the exact same order. "
            "The candidate must appear with all its tokens consecutive and in order - no gaps, no extra tokens in between. "
            "Answer {0} if the candidate appears as a contiguous subsequence, {1} if it does not, or {2} if you are unsure. Please put your answer in curly braces, e.g, {2}."
        )

    def process_native_binary_question(
        self,
        model: Any,
        question: QuestionEntry,
        video_index: int,
        max_tokens: int = 700,
        max_frames: int = 256,
        current_video_time: Optional[float] = 0.0,
        include_uncertain: bool = True,
    ) -> Dict[str, Any]:
        """Ask one native binary question and score it ('0' yes, '1' no, '2' unsure)."""
        if not question.is_native_binary:
            raise ValueError("process_native_binary_question requires native binary format question")

        question_text = self.build_native_binary_question_text(question, include_uncertain)

        if self.verbose:
            self._print_native_binary_question_details(question, video_index, question_text)

        effective_time = current_video_time
        if effective_time is None:
            effective_time = float(question.question_time)

        response = model.ask_question(
            question_text,
            current_video_time=effective_time,
            max_tokens=max_tokens,
            max_frames_in_video=max_frames,
        )

        video_truncated = None
        if hasattr(model, 'was_video_truncated'):
            video_truncated = model.was_video_truncated()

        if self.verbose:
            self._print_response(response)

        raw_answer = self.extract_answer(response)
        predicted_answer = self.normalize_native_binary_answer(raw_answer)

        raw_correct = (question.binary_answer or "").strip().lower()
        if raw_correct == 'yes':
            correct_answer = '0'
        elif raw_correct == 'no':
            correct_answer = '1'
        else:
            correct_answer = raw_correct

        is_correct = predicted_answer == correct_answer
        is_dont_know = predicted_answer == '2'

        question_identifier = getattr(question, 'question_order', question.question_id)
        candidate_path = question.candidate.clip_path if question.candidate else "N/A"
        candidate_name = os.path.basename(candidate_path) if candidate_path else "N/A"

        status_icon = 'CORRECT' if is_correct else ('UNCERTAIN' if is_dont_know else 'INCORRECT')
        print(
            f"Q{question_identifier}: {candidate_name} -> "
            f"Predicted: {predicted_answer.upper() or 'UNKNOWN'}, "
            f"Correct: {correct_answer.upper() or 'UNKNOWN'} - {status_icon}"
        )

        has_answer = bool(predicted_answer)

        if hasattr(model, "record_question_outcome"):
            model.record_question_outcome(
                current_video_time=effective_time,
                is_correct=is_correct,
                is_dont_know=is_dont_know,
                is_answered=has_answer,
            )

        if self.verbose:
            if is_dont_know:
                print("Result: UNCERTAIN")
            else:
                print(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")
            print(f"{'='*60}")

        _free_memory()

        result = {
            'video_index': video_index,
            'question_id': question.question_id,
            'question_order': getattr(question, 'question_order', None),
            'predicted': predicted_answer,
            'correct': correct_answer,
            'is_correct': is_correct,
            'is_dont_know': is_dont_know,
            'num_options': 3 if include_uncertain else 2,
            'response': response,
            'saw_all_frames': None if video_truncated is None else not video_truncated,
            'max_tokens': max_tokens,
            'max_frames': max_frames,
            'is_native_binary': True,
            'candidate_present': question.candidate.present if question.candidate else None,
            'candidate_clip_path': candidate_path,
        }

        return result

    def _print_native_binary_question_details(
        self,
        question: QuestionEntry,
        video_index: int,
        question_text: str,
    ) -> None:
        print(f"\n{'='*60}")
        print("NATIVE BINARY QUESTION DETAILS")
        print(f"{'='*60}")
        question_number = getattr(question, 'question_order', question.question_id)
        print(f"Video Index: {video_index}")
        print(f"Question #: {question_number} (ID: {question.question_id})")
        print(f"Mode: {question.question_mode or 'exists'}")
        print(f"Expected Answer: {question.binary_answer}")

        if question.candidate:
            print("\nCandidate:")
            print(f"  Clip: {os.path.basename(question.candidate.clip_path)}")
            print(f"  Present in video: {question.candidate.present}")
            if question.candidate.sequence:
                seq_display = ", ".join(question.candidate.sequence[:5])
                if len(question.candidate.sequence) > 5:
                    seq_display += f", ... ({len(question.candidate.sequence)} total)"
                print(f"  Sequence: {seq_display}")

        print("\nFull Question Text:")
        print(question_text)
        print(f"{'='*60}")


def _free_memory() -> None:
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
