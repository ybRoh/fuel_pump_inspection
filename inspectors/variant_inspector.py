"""
이종품 검사 모듈
Variant Inspection Module for Fuel Pump

검사 항목:
- 부품 형상 일치 여부
- 부품 색상 일치 여부
- 정품/이종품 판별
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import os


@dataclass
class VariantResult:
    """이종품 검사 결과 데이터 클래스"""
    is_correct: bool
    detected_part: Optional[str]
    expected_part: Optional[str]
    similarity: float
    color_match: bool
    shape_match: bool
    message: str

    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            'is_correct': self.is_correct,
            'detected_part': self.detected_part,
            'expected_part': self.expected_part,
            'similarity': self.similarity,
            'color_match': self.color_match,
            'shape_match': self.shape_match,
            'message': self.message
        }


class VariantInspector:
    """이종품 검사 클래스"""

    def __init__(self, config: dict, valid_parts: List[dict]):
        """
        Args:
            config: 검사 설정 딕셔너리
                - similarity_threshold: 유사도 임계값
                - color_tolerance: 색상 허용 오차
                - shape_tolerance: 형상 허용 오차
            valid_parts: 등록된 정품 부품 목록
        """
        self.config = config
        self.valid_parts = valid_parts
        self.similarity_threshold = config.get('similarity_threshold', 0.85)
        self.color_tolerance = config.get('color_tolerance', 20)
        self.shape_tolerance = config.get('shape_tolerance', 0.3)

        # 부품별 마스터 이미지
        self.part_masters: Dict[str, np.ndarray] = {}
        self.part_features: Dict[str, Tuple] = {}

        # 특징점 검출기
        self.orb = cv2.ORB_create(nfeatures=2000)

    def load_masters(self, master_dir: str) -> None:
        """
        부품별 마스터 이미지 로드

        Args:
            master_dir: 마스터 이미지 디렉토리 경로
        """
        for part in self.valid_parts:
            part_id = part['id']

            # 여러 확장자 시도
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                master_path = os.path.join(master_dir, f"{part_id}{ext}")
                if os.path.exists(master_path):
                    img = cv2.imread(master_path)
                    if img is not None:
                        self.part_masters[part_id] = img

                        # 특징점 추출
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        kp, desc = self.orb.detectAndCompute(gray, None)
                        self.part_features[part_id] = (kp, desc)

                        print(f"마스터 로드: {part_id}")
                    break

    def register_master(self, part_id: str, image: np.ndarray) -> None:
        """
        새 마스터 이미지 등록

        Args:
            part_id: 부품 ID
            image: BGR 형식의 마스터 이미지
        """
        self.part_masters[part_id] = image.copy()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, desc = self.orb.detectAndCompute(gray, None)
        self.part_features[part_id] = (kp, desc)

        print(f"마스터 등록 완료: {part_id}")

    def save_master(self, part_id: str, save_dir: str) -> bool:
        """
        마스터 이미지 저장

        Args:
            part_id: 부품 ID
            save_dir: 저장 디렉토리

        Returns:
            저장 성공 여부
        """
        if part_id not in self.part_masters:
            return False

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{part_id}.jpg")
        cv2.imwrite(save_path, self.part_masters[part_id])

        return True

    def inspect(self, image: np.ndarray,
                expected_part_id: str = None) -> Tuple[bool, VariantResult]:
        """
        이종품 검사 수행

        Args:
            image: BGR 형식의 검사 이미지
            expected_part_id: 예상 부품 ID (선택)

        Returns:
            (합격 여부, 이종품 검사 결과)
        """
        if not self.part_masters:
            return True, VariantResult(
                is_correct=True,
                detected_part=None,
                expected_part=expected_part_id,
                similarity=0,
                color_match=True,
                shape_match=True,
                message="마스터 이미지 미등록 - 검사 생략"
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, desc = self.orb.detectAndCompute(gray, None)

        if desc is None:
            return False, VariantResult(
                is_correct=False,
                detected_part=None,
                expected_part=expected_part_id,
                similarity=0,
                color_match=False,
                shape_match=False,
                message="부품 인식 불가"
            )

        # 1. 모든 마스터와 매칭하여 가장 유사한 부품 찾기
        best_match = None
        best_similarity = 0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        for part_id, (ref_kp, ref_desc) in self.part_features.items():
            if ref_desc is None:
                continue

            try:
                matches = bf.match(ref_desc, desc)
                matches = sorted(matches, key=lambda x: x.distance)

                # 유사도 계산
                good_matches = [m for m in matches if m.distance < 50]
                similarity = len(good_matches) / max(len(ref_kp), 1)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = part_id
            except cv2.error:
                continue

        # 2. 색상 검사
        color_match = self._check_color(image, best_match)

        # 3. 형상 검사 (윤곽선 비교)
        shape_match = self._check_shape(gray, best_match)

        # 4. 판정
        detected_part = best_match if best_similarity > self.similarity_threshold else None

        if expected_part_id:
            is_correct = (detected_part == expected_part_id and
                         color_match and shape_match)
        else:
            is_correct = (detected_part is not None and
                         color_match and shape_match)

        # 메시지 생성
        message = self._generate_message(
            is_correct, detected_part, expected_part_id,
            best_similarity, color_match, shape_match
        )

        return is_correct, VariantResult(
            is_correct=is_correct,
            detected_part=detected_part,
            expected_part=expected_part_id,
            similarity=best_similarity,
            color_match=color_match,
            shape_match=shape_match,
            message=message
        )

    def _check_color(self, image: np.ndarray, part_id: str) -> bool:
        """
        색상 일치 확인

        Args:
            image: BGR 형식 이미지
            part_id: 비교할 부품 ID

        Returns:
            색상 일치 여부
        """
        if part_id not in self.part_masters:
            return True

        master = self.part_masters[part_id]

        # 크기 맞추기
        if image.shape[:2] != master.shape[:2]:
            image = cv2.resize(image, (master.shape[1], master.shape[0]))

        # HSV 히스토그램 비교
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_master = cv2.cvtColor(master, cv2.COLOR_BGR2HSV)

        hist_img = cv2.calcHist([hsv_img], [0, 1], None, [50, 60],
                                [0, 180, 0, 256])
        hist_master = cv2.calcHist([hsv_master], [0, 1], None, [50, 60],
                                   [0, 180, 0, 256])

        cv2.normalize(hist_img, hist_img, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_master, hist_master, 0, 1, cv2.NORM_MINMAX)

        correlation = cv2.compareHist(hist_img, hist_master,
                                      cv2.HISTCMP_CORREL)

        return correlation > 0.8

    def _check_shape(self, gray: np.ndarray, part_id: str) -> bool:
        """
        형상 일치 확인

        Args:
            gray: 그레이스케일 이미지
            part_id: 비교할 부품 ID

        Returns:
            형상 일치 여부
        """
        if part_id not in self.part_masters:
            return True

        master = self.part_masters[part_id]
        master_gray = cv2.cvtColor(master, cv2.COLOR_BGR2GRAY)

        # 크기 맞추기
        if gray.shape != master_gray.shape:
            gray = cv2.resize(gray, (master_gray.shape[1], master_gray.shape[0]))

        # 윤곽선 추출
        _, thresh_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        _, thresh_master = cv2.threshold(master_gray, 127, 255, cv2.THRESH_BINARY)

        contours_img, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
        contours_master, _ = cv2.findContours(thresh_master, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)

        if not contours_img or not contours_master:
            return False

        # 가장 큰 윤곽선 비교
        cnt_img = max(contours_img, key=cv2.contourArea)
        cnt_master = max(contours_master, key=cv2.contourArea)

        match_score = cv2.matchShapes(cnt_img, cnt_master,
                                      cv2.CONTOURS_MATCH_I2, 0)

        return match_score < self.shape_tolerance

    def _generate_message(self, is_correct: bool, detected: Optional[str],
                         expected: Optional[str], similarity: float,
                         color_ok: bool, shape_ok: bool) -> str:
        """
        검사 결과 메시지 생성

        Args:
            is_correct: 합격 여부
            detected: 검출된 부품 ID
            expected: 예상 부품 ID
            similarity: 유사도
            color_ok: 색상 일치 여부
            shape_ok: 형상 일치 여부

        Returns:
            결과 메시지 문자열
        """
        if is_correct:
            return "정상 - 정품 확인"

        messages = []

        if detected != expected:
            detected_str = detected or '미확인'
            expected_str = expected or '미지정'
            messages.append(f"이종품 검출: {detected_str} (예상: {expected_str})")

        if not color_ok:
            messages.append("색상 불일치")

        if not shape_ok:
            messages.append("형상 불일치")

        if similarity < self.similarity_threshold:
            messages.append(f"유사도 부족: {similarity:.1%}")

        return " / ".join(messages) if messages else "불량"

    def get_part_info(self, part_id: str) -> Optional[dict]:
        """
        부품 정보 조회

        Args:
            part_id: 부품 ID

        Returns:
            부품 정보 딕셔너리 또는 None
        """
        for part in self.valid_parts:
            if part['id'] == part_id:
                return part
        return None

    def visualize_result(self, image: np.ndarray,
                         result: VariantResult) -> np.ndarray:
        """
        이종품 검사 결과 시각화

        Args:
            image: 원본 이미지
            result: 이종품 검사 결과

        Returns:
            시각화된 이미지
        """
        display = image.copy()

        # 결과 표시
        color = (0, 255, 0) if result.is_correct else (0, 0, 255)
        status = "OK" if result.is_correct else "NG"

        cv2.putText(display, f"Variant: {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        if result.detected_part:
            cv2.putText(display, f"Detected: {result.detected_part}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(display, f"Similarity: {result.similarity:.1%}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 색상/형상 상태
        color_status = "O" if result.color_match else "X"
        shape_status = "O" if result.shape_match else "X"
        cv2.putText(display, f"Color: {color_status} / Shape: {shape_status}",
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return display
