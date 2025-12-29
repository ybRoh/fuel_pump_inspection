#!/usr/bin/env python3
"""
딥러닝 학습용 데이터 수집 스크립트
Data Collection Script for Deep Learning Training
"""

import cv2
import os
import sys
import time
from datetime import datetime
import argparse

# 카메라 설정
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False


class DataCollector:
    """데이터 수집 클래스"""

    # 결함 유형 목록
    DEFECT_TYPES = [
        'normal',        # 정상
        'scratch',       # 스크래치
        'dent',          # 찍힘
        'crack',         # 균열
        'contamination', # 이물질
        'burr',          # 버
        'missing_part',  # 부품 누락
    ]

    def __init__(self, output_dir: str = 'datasets/collected'):
        self.output_dir = output_dir
        self.camera = None
        self.current_defect = 'normal'
        self.count = {}

        # 출력 디렉토리 생성
        self._create_directories()

        # 기존 이미지 수 카운트
        self._count_existing_images()

        # 카메라 초기화
        self._init_camera()

    def _create_directories(self):
        """결함 유형별 디렉토리 생성"""
        for defect_type in self.DEFECT_TYPES:
            dir_path = os.path.join(self.output_dir, defect_type)
            os.makedirs(dir_path, exist_ok=True)
        print(f"데이터 저장 경로: {self.output_dir}")

    def _count_existing_images(self):
        """기존 이미지 수 카운트"""
        for defect_type in self.DEFECT_TYPES:
            dir_path = os.path.join(self.output_dir, defect_type)
            if os.path.exists(dir_path):
                count = len([f for f in os.listdir(dir_path)
                           if f.endswith(('.jpg', '.png'))])
                self.count[defect_type] = count
            else:
                self.count[defect_type] = 0

    def _init_camera(self):
        """카메라 초기화"""
        if PICAMERA_AVAILABLE:
            self.camera = Picamera2()
            config = self.camera.create_still_configuration(
                main={"size": (1920, 1080)}
            )
            self.camera.configure(config)
            self.camera.start()
            print("Pi Camera 초기화 완료")
        else:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            print("USB Camera 초기화 완료")

    def capture(self):
        """이미지 캡처"""
        if PICAMERA_AVAILABLE:
            frame = self.camera.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = self.camera.read()
            if not ret:
                return None
        return frame

    def save_image(self, frame, defect_type: str):
        """이미지 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{defect_type}_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, defect_type, filename)

        cv2.imwrite(filepath, frame)
        self.count[defect_type] += 1

        print(f"저장: {filepath}")
        print(f"  {defect_type} 총 {self.count[defect_type]}장")

        return filepath

    def print_status(self):
        """현재 수집 현황 출력"""
        print("\n" + "="*50)
        print("데이터 수집 현황")
        print("="*50)
        total = 0
        for defect_type in self.DEFECT_TYPES:
            count = self.count.get(defect_type, 0)
            total += count
            bar = "█" * (count // 10) + "░" * (10 - count // 10)
            print(f"  {defect_type:15} [{bar}] {count:4}장")
        print("-"*50)
        print(f"  {'총계':15} {' '*12} {total:4}장")
        print("="*50 + "\n")

    def run_interactive(self):
        """인터랙티브 데이터 수집 모드"""
        print("\n" + "#"*50)
        print("# 딥러닝 학습 데이터 수집 모드")
        print("#"*50)
        print("\n[단축키]")
        print("  0: 정상 (normal)")
        print("  1: 스크래치 (scratch)")
        print("  2: 찍힘 (dent)")
        print("  3: 균열 (crack)")
        print("  4: 이물질 (contamination)")
        print("  5: 버 (burr)")
        print("  6: 부품누락 (missing_part)")
        print("  SPACE: 현재 선택된 유형으로 저장")
        print("  s: 현황 보기")
        print("  q: 종료")
        print()

        self.print_status()
        print(f"현재 선택: [{self.current_defect}]")

        while True:
            frame = self.capture()
            if frame is None:
                continue

            # 화면에 정보 표시
            display = frame.copy()

            # 현재 선택된 결함 유형
            cv2.putText(display, f"Type: {self.current_defect}",
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 저장된 이미지 수
            count = self.count.get(self.current_defect, 0)
            cv2.putText(display, f"Count: {count}",
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 안내 메시지
            cv2.putText(display, "SPACE: Save | 0-6: Change Type | q: Quit",
                       (20, display.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Data Collection", display)

            key = cv2.waitKey(1) & 0xFF

            # 숫자키로 결함 유형 선택
            if ord('0') <= key <= ord('6'):
                idx = key - ord('0')
                self.current_defect = self.DEFECT_TYPES[idx]
                print(f"\n결함 유형 변경: {self.current_defect}")

            # 스페이스바로 저장
            elif key == ord(' '):
                self.save_image(frame, self.current_defect)

            # 's'로 현황 보기
            elif key == ord('s'):
                self.print_status()

            # 'q'로 종료
            elif key == ord('q'):
                break

        self.cleanup()
        self.print_status()
        print("데이터 수집 종료")

    def run_burst(self, defect_type: str, count: int, interval: float = 0.5):
        """연속 촬영 모드"""
        print(f"\n연속 촬영 모드: {defect_type} x {count}장")
        print(f"촬영 간격: {interval}초")
        print("3초 후 시작...")
        time.sleep(3)

        for i in range(count):
            frame = self.capture()
            if frame is not None:
                self.save_image(frame, defect_type)
                print(f"  {i+1}/{count} 완료")
            time.sleep(interval)

        print(f"\n연속 촬영 완료: {count}장")
        self.cleanup()

    def cleanup(self):
        """리소스 정리"""
        cv2.destroyAllWindows()
        if PICAMERA_AVAILABLE:
            self.camera.stop()
        else:
            self.camera.release()


def main():
    parser = argparse.ArgumentParser(description="딥러닝 학습 데이터 수집")
    parser.add_argument('--output', '-o', default='datasets/collected',
                       help='저장 디렉토리 (기본: datasets/collected)')
    parser.add_argument('--mode', '-m', choices=['interactive', 'burst'],
                       default='interactive', help='수집 모드')
    parser.add_argument('--type', '-t', default='normal',
                       help='결함 유형 (burst 모드용)')
    parser.add_argument('--count', '-c', type=int, default=10,
                       help='촬영 수량 (burst 모드용)')
    parser.add_argument('--interval', '-i', type=float, default=0.5,
                       help='촬영 간격 초 (burst 모드용)')

    args = parser.parse_args()

    collector = DataCollector(args.output)

    if args.mode == 'interactive':
        collector.run_interactive()
    elif args.mode == 'burst':
        collector.run_burst(args.type, args.count, args.interval)


if __name__ == "__main__":
    main()
