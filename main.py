#!/usr/bin/env python3
"""
연료펌프 모듈 검사 시스템 메인 모듈
Fuel Pump Module Inspection System - Main Entry Point

기능:
- 외관 결함 검사 (스크래치, 찍힘, 균열, 이물질, 사출불량)
- 바코드 부착 검사
- 이종품 검사

사용법:
    python main.py --mode run --trigger manual
    python main.py --mode register --part-id FP-2024-A01
"""

import cv2
import numpy as np
import yaml
import time
import os
import json
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Optional, List

# 검사 모듈
from inspectors import (
    DefectInspector, DefectType, Defect,
    BarcodeInspector, BarcodeResult,
    VariantInspector, VariantResult
)
from inspectors.deep_learning_inspector import DeepLearningInspector, DLInspectionResult

# 유틸리티 모듈
from utils import CameraManager, GPIOController, DatabaseManager


@dataclass
class InspectionResult:
    """검사 결과 데이터 클래스"""
    timestamp: str
    part_id: str
    barcode: Optional[str]

    # 개별 검사 결과
    defect_ok: bool
    barcode_ok: bool
    variant_ok: bool
    dl_ok: bool = True  # 딥러닝 검사 결과

    # 상세 정보
    defects: List[dict] = field(default_factory=list)
    barcode_result: dict = field(default_factory=dict)
    variant_result: dict = field(default_factory=dict)
    dl_result: dict = field(default_factory=dict)  # 딥러닝 검사 상세

    # 최종 판정
    final_result: str = "OK"  # "OK" or "NG"
    ng_reason: str = ""

    # 검사 시간
    inspection_time_ms: float = 0.0
    dl_inference_time_ms: float = 0.0  # 딥러닝 추론 시간


class FuelPumpInspectionSystem:
    """연료펌프 모듈 검사 시스템 클래스"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        print("=" * 50)
        print("연료펌프 모듈 검사 시스템 초기화")
        print("=" * 50)

        # 설정 로드
        self.config = self._load_config(config_path)

        # 카메라 초기화
        self._init_camera()

        # 검사 모듈 초기화
        self._init_inspectors()

        # GPIO 초기화 (PLC 연동)
        self._init_gpio()

        # 데이터베이스 초기화
        self._init_database()

        # 결과 저장 디렉토리
        self.log_dir = self.config.get('logging', {}).get(
            'log_dir', 'logs/inspection_results'
        )
        os.makedirs(self.log_dir, exist_ok=True)

        # 통계
        self.stats = {
            'total': 0,
            'ok': 0,
            'ng': 0,
            'ng_reasons': {}
        }

        print("\n시스템 초기화 완료!")
        print("=" * 50)

    def _load_config(self, config_path: str) -> dict:
        """설정 파일 로드"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"설정 로드: {config_path}")
            return config
        else:
            print(f"설정 파일 없음, 기본값 사용: {config_path}")
            return self._default_config()

    def _default_config(self) -> dict:
        """기본 설정 반환"""
        return {
            'system': {'name': '연료펌프 모듈 검사기', 'line': 'LINE-01'},
            'camera': {
                'top': {'id': 0, 'resolution': [1920, 1080], 'exposure': 10000}
            },
            'inspection': {
                'defect': {'enabled': True, 'min_defect_area': 50},
                'barcode': {'enabled': True, 'required_prefix': 'FP'},
                'variant': {'enabled': True, 'similarity_threshold': 0.85}
            },
            'valid_parts': [],
            'gpio': {'enabled': False},
            'database': {'enabled': True, 'type': 'sqlite', 'path': 'logs/inspection.db'},
            'logging': {'save_images': True, 'log_dir': 'logs/inspection_results'}
        }

    def _init_camera(self) -> None:
        """카메라 초기화"""
        print("\n[카메라 초기화]")

        self.camera_manager = CameraManager(self.config.get('camera', {}))

        # 상단 카메라 추가
        camera_config = self.config.get('camera', {}).get('top', {})
        try:
            self.camera_manager.add_camera('top', camera_config, use_picamera=True)
            self.camera_manager.start_all()
        except Exception as e:
            print(f"Pi Camera 실패, USB 카메라 시도: {e}")
            try:
                self.camera_manager.add_camera('top', camera_config, use_picamera=False)
                self.camera_manager.start_all()
            except Exception as e2:
                print(f"카메라 초기화 실패: {e2}")

    def _init_inspectors(self) -> None:
        """검사 모듈 초기화"""
        print("\n[검사 모듈 초기화]")

        inspection_config = self.config.get('inspection', {})

        # 외관 결함 검사
        self.defect_inspector = DefectInspector(
            inspection_config.get('defect', {})
        )
        print("  - 외관 결함 검사 모듈: OK")

        # 바코드 검사
        self.barcode_inspector = BarcodeInspector(
            inspection_config.get('barcode', {})
        )
        print("  - 바코드 검사 모듈: OK")

        # 이종품 검사
        self.variant_inspector = VariantInspector(
            inspection_config.get('variant', {}),
            self.config.get('valid_parts', [])
        )

        # 마스터 이미지 로드
        master_dir = 'reference/part_masters'
        if os.path.exists(master_dir):
            self.variant_inspector.load_masters(master_dir)
        print("  - 이종품 검사 모듈: OK")

        # 딥러닝 검사 모듈
        dl_config = self.config.get('deep_learning', {})
        self.use_deep_learning = dl_config.get('enabled', False)
        self.dl_only_mode = dl_config.get('use_instead_of_classic', False)

        if self.use_deep_learning:
            self.dl_inspector = DeepLearningInspector(dl_config)
            if self.dl_inspector.is_available:
                print("  - 딥러닝 검사 모듈: OK")
            else:
                print("  - 딥러닝 검사 모듈: 모델 없음 (기존 방식 사용)")
                self.use_deep_learning = False
        else:
            self.dl_inspector = None
            print("  - 딥러닝 검사 모듈: 비활성화")

    def _init_gpio(self) -> None:
        """GPIO 초기화"""
        print("\n[GPIO 초기화]")

        gpio_config = self.config.get('gpio', {})
        self.gpio = GPIOController(gpio_config)

        if self.gpio.initialize():
            print("  - GPIO 초기화: OK")
        else:
            print("  - GPIO 초기화: 비활성화")

    def _init_database(self) -> None:
        """데이터베이스 초기화"""
        print("\n[데이터베이스 초기화]")

        db_config = self.config.get('database', {})
        self.database = DatabaseManager(db_config)

        if self.database.initialize():
            print("  - 데이터베이스 초기화: OK")
        else:
            print("  - 데이터베이스 초기화: 비활성화")

    def capture(self) -> np.ndarray:
        """이미지 캡처"""
        try:
            return self.camera_manager.capture('top')
        except Exception as e:
            print(f"캡처 오류: {e}")
            # 테스트용 더미 이미지
            return np.zeros((1080, 1920, 3), dtype=np.uint8)

    def inspect(self, expected_part_id: str = None) -> InspectionResult:
        """
        전체 검사 수행

        Args:
            expected_part_id: 예상 부품 ID

        Returns:
            검사 결과
        """
        start_time = time.time()

        # 이미지 캡처
        frame = self.capture()

        # 1. 바코드 검사
        barcode_ok, barcode_result = self.barcode_inspector.inspect(frame)

        # 바코드에서 부품 ID 추출
        part_id = expected_part_id
        if barcode_result.data:
            part_id = barcode_result.data

        # 2. 이종품 검사
        variant_ok, variant_result = self.variant_inspector.inspect(
            frame, expected_part_id
        )

        # 3. 외관 결함 검사 (기존 방식)
        defect_ok = True
        defects = []

        if not self.dl_only_mode:
            # 기존 OpenCV 기반 검사
            defect_ok, defects = self.defect_inspector.inspect(frame)

        # 4. 딥러닝 검사
        dl_ok = True
        dl_result = None
        dl_inference_time = 0.0

        if self.use_deep_learning and self.dl_inspector is not None:
            dl_result = self.dl_inspector.inspect(frame)
            dl_ok = dl_result.is_ok
            dl_inference_time = dl_result.inference_time_ms

            # 딥러닝 결함을 기존 결함 목록에 추가
            for det in dl_result.detections:
                defects.append(det.to_dict())

            # 딥러닝 전용 모드일 경우 딥러닝 결과로 대체
            if self.dl_only_mode:
                defect_ok = dl_ok

        # 검사 시간 계산
        inspection_time = (time.time() - start_time) * 1000

        # 최종 판정 (모든 검사 통과해야 OK)
        all_ok = defect_ok and barcode_ok and variant_ok and dl_ok
        final_result = "OK" if all_ok else "NG"

        # NG 사유 생성
        ng_reasons = []
        if not defect_ok and not self.dl_only_mode:
            defect_types = set(d.type.value if hasattr(d, 'type') else d.get('type', '') for d in defects if hasattr(d, 'type') or isinstance(d, dict))
            if defect_types:
                ng_reasons.append(f"외관불량({','.join(defect_types)})")
        if not dl_ok and dl_result:
            ng_reasons.append(f"[DL]{dl_result.message}")
        if not barcode_ok:
            ng_reasons.append(f"바코드불량({barcode_result.message})")
        if not variant_ok:
            ng_reasons.append(f"이종품({variant_result.message})")

        ng_reason = " / ".join(ng_reasons) if ng_reasons else ""

        # 결과 생성
        result = InspectionResult(
            timestamp=datetime.now().isoformat(),
            part_id=part_id or "UNKNOWN",
            barcode=barcode_result.data,
            defect_ok=defect_ok,
            barcode_ok=barcode_ok,
            variant_ok=variant_ok,
            dl_ok=dl_ok,
            defects=[d.to_dict() if hasattr(d, 'to_dict') else d for d in defects],
            barcode_result=barcode_result.to_dict(),
            variant_result=variant_result.to_dict(),
            dl_result=dl_result.to_dict() if dl_result else {},
            final_result=final_result,
            ng_reason=ng_reason,
            inspection_time_ms=inspection_time,
            dl_inference_time_ms=dl_inference_time
        )

        # 통계 업데이트
        self._update_stats(result)

        # 결과 출력
        self._output_result(result, frame, dl_result)

        # 결과 저장
        self._save_result(result, frame)

        return result

    def _update_stats(self, result: InspectionResult) -> None:
        """통계 업데이트"""
        self.stats['total'] += 1

        if result.final_result == "OK":
            self.stats['ok'] += 1
        else:
            self.stats['ng'] += 1
            reason = result.ng_reason.split('/')[0].strip() if result.ng_reason else "기타"
            self.stats['ng_reasons'][reason] = \
                self.stats['ng_reasons'].get(reason, 0) + 1

    def _output_result(self, result: InspectionResult, frame: np.ndarray,
                       dl_result: DLInspectionResult = None) -> None:
        """결과 출력 (GPIO + 화면)"""
        # GPIO 출력
        self.gpio.output_result(result.final_result == "OK")

        # 화면 표시
        display = frame.copy()

        # 딥러닝 결과 시각화
        if dl_result and self.use_deep_learning and self.dl_inspector:
            display = self.dl_inspector.visualize(display, dl_result)

        # 기존 결함 표시 (딥러닝 전용 모드가 아닌 경우)
        if not self.dl_only_mode:
            for defect in result.defects:
                if 'bbox' in defect:
                    x, y, w, h = defect['bbox']
                    cv2.rectangle(display, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    label = f"{defect.get('type', 'defect')}"
                    cv2.putText(display, label, (x, y-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 판정 결과 표시
        color = (0, 255, 0) if result.final_result == "OK" else (0, 0, 255)
        cv2.putText(display, result.final_result, (50, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)

        if result.ng_reason:
            # NG 사유 (여러 줄로 분리)
            reasons = result.ng_reason.split(' / ')
            for i, reason in enumerate(reasons[:3]):  # 최대 3줄
                cv2.putText(display, reason, (50, 140 + i*30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 바코드 정보
        y_offset = 140 + min(len(result.ng_reason.split(' / ')), 3) * 30 + 20
        if result.barcode:
            cv2.putText(display, f"ID: {result.barcode}", (50, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_offset += 30

        # 검사 시간
        time_text = f"Total: {result.inspection_time_ms:.1f}ms"
        if result.dl_inference_time_ms > 0:
            time_text += f" | DL: {result.dl_inference_time_ms:.1f}ms"
        cv2.putText(display, time_text, (50, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # 검사 모드 표시
        mode_text = "[DL]" if self.use_deep_learning else "[Classic]"
        if self.dl_only_mode:
            mode_text = "[DL Only]"
        cv2.putText(display, mode_text, (display.shape[1] - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

        cv2.imshow("Fuel Pump Inspection", display)

    def _save_result(self, result: InspectionResult, frame: np.ndarray) -> None:
        """결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # 이미지 저장 경로
        image_path = None
        save_images = self.config.get('logging', {}).get('save_images', True)

        if result.final_result == "NG" and save_images:
            image_path = os.path.join(self.log_dir, f"{timestamp}_NG.jpg")
            cv2.imwrite(image_path, frame)

        # JSON 로그
        log_path = os.path.join(self.log_dir, f"{timestamp}.json")
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)

        # 데이터베이스 저장
        if self.database.is_available:
            self.database.save_result(result, image_path)

    def register_reference(self, part_id: str = None) -> None:
        """기준 이미지 등록 모드"""
        print("\n" + "=" * 50)
        print("기준 이미지 등록 모드")
        print("=" * 50)
        print("정상 제품을 배치하고 Enter를 누르세요")
        print("종료: 'q' 키")
        print()

        while True:
            frame = self.capture()

            # 안내 텍스트
            display = frame.copy()
            cv2.putText(display, "Press ENTER to register", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(display, "Press 'q' to exit", (50, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            if part_id:
                cv2.putText(display, f"Part ID: {part_id}", (50, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Register Reference", display)

            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # Enter
                # 결함 검사 기준 등록
                self.defect_inspector.set_reference(frame)
                print("외관 검사 기준 등록 완료")

                # 이종품 검사 기준 등록
                if part_id:
                    self.variant_inspector.register_master(part_id, frame)

                    # 마스터 이미지 저장
                    master_dir = 'reference/part_masters'
                    os.makedirs(master_dir, exist_ok=True)
                    self.variant_inspector.save_master(part_id, master_dir)
                    print(f"부품 '{part_id}' 마스터 등록 및 저장 완료")

                print("기준 이미지 등록 완료!")
                self.gpio.buzzer_beep(0.1, 2)

            elif key == ord('q'):
                break

        cv2.destroyAllWindows()

    def run_continuous(self, trigger_mode: str = "manual") -> None:
        """
        연속 검사 모드

        Args:
            trigger_mode: 트리거 모드 ('manual', 'gpio', 'auto')
        """
        print("\n" + "=" * 50)
        print("연속 검사 모드 시작")
        print("=" * 50)
        print(f"트리거 모드: {trigger_mode}")
        print("'스페이스': 검사 | 'r': 기준 등록 | 's': 통계 | 'q': 종료")
        print()

        while True:
            trigger = False

            if trigger_mode == "manual":
                frame = self.capture()
                cv2.imshow("Fuel Pump Inspection", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord(' '):  # 스페이스바
                    trigger = True
                elif key == ord('r'):
                    self.register_reference()
                elif key == ord('s'):
                    self._print_stats()
                elif key == ord('q'):
                    break

            elif trigger_mode == "gpio":
                if self.gpio.is_available:
                    trigger = self.gpio.read_trigger()
                    if trigger:
                        time.sleep(0.1)  # 디바운스
                else:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('q'):
                        break

            elif trigger_mode == "auto":
                trigger = True
                time.sleep(1)  # 1초 간격

            if trigger:
                result = self.inspect()
                status = "OK" if result.final_result == "OK" else "NG"
                print(f"[{result.timestamp}] {result.part_id}: "
                      f"{status} ({result.inspection_time_ms:.1f}ms)")

                if result.final_result == "NG":
                    print(f"  └─ {result.ng_reason}")

        self._cleanup()

    def _print_stats(self) -> None:
        """통계 출력"""
        print("\n" + "=" * 50)
        print("검사 통계")
        print("=" * 50)
        print(f"총 검사: {self.stats['total']}")

        total = max(self.stats['total'], 1)
        ok_rate = self.stats['ok'] / total * 100
        ng_rate = self.stats['ng'] / total * 100

        print(f"양품(OK): {self.stats['ok']} ({ok_rate:.1f}%)")
        print(f"불량(NG): {self.stats['ng']} ({ng_rate:.1f}%)")

        if self.stats['ng_reasons']:
            print("\nNG 사유별:")
            for reason, count in sorted(
                self.stats['ng_reasons'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                print(f"  - {reason}: {count}건")

        print("=" * 50 + "\n")

    def _cleanup(self) -> None:
        """종료 처리"""
        print("\n시스템 종료 중...")

        cv2.destroyAllWindows()

        self.camera_manager.stop_all()
        self.gpio.cleanup()
        self.database.close()

        self._print_stats()
        print("시스템 종료 완료")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="연료펌프 모듈 검사 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  python main.py --mode run --trigger manual    # 수동 검사 모드
  python main.py --mode run --trigger gpio      # GPIO 트리거 검사
  python main.py --mode run --trigger auto      # 자동 연속 검사
  python main.py --mode register --part-id FP-2024-A01  # 기준 등록
        """
    )

    parser.add_argument(
        '--mode',
        choices=['run', 'register'],
        default='run',
        help='실행 모드 (run: 검사, register: 기준 등록)'
    )

    parser.add_argument(
        '--trigger',
        choices=['manual', 'gpio', 'auto'],
        default='manual',
        help='트리거 모드 (manual: 수동, gpio: GPIO, auto: 자동)'
    )

    parser.add_argument(
        '--part-id',
        type=str,
        help='등록할 부품 ID (register 모드에서 사용)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='설정 파일 경로'
    )

    args = parser.parse_args()

    # 시스템 초기화
    system = FuelPumpInspectionSystem(args.config)

    # 모드별 실행
    if args.mode == 'register':
        system.register_reference(args.part_id)
    else:
        system.run_continuous(args.trigger)


if __name__ == "__main__":
    main()
