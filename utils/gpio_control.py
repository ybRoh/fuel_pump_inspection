"""
GPIO 제어 모듈
GPIO Control Module for PLC Communication

기능:
- 검사 트리거 신호 입력
- OK/NG 결과 신호 출력
- 경보음(버저) 제어
"""

import time
from typing import Callable, Optional

# GPIO 라이브러리 지원 확인
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False


class GPIOController:
    """GPIO 제어 클래스"""

    def __init__(self, config: dict):
        """
        Args:
            config: GPIO 설정 딕셔너리
                - enabled: GPIO 사용 여부
                - trigger_input: 트리거 입력 핀
                - ok_output: OK 출력 핀
                - ng_output: NG 출력 핀
                - buzzer: 버저 핀
                - signal_duration: 신호 유지 시간
        """
        self.config = config
        self.enabled = config.get('enabled', True) and GPIO_AVAILABLE

        # 핀 번호
        self.trigger_pin = config.get('trigger_input', 17)
        self.ok_pin = config.get('ok_output', 27)
        self.ng_pin = config.get('ng_output', 22)
        self.buzzer_pin = config.get('buzzer', 23)

        # 신호 유지 시간
        self.signal_duration = config.get('signal_duration', 0.5)

        # 콜백 함수
        self._trigger_callback: Optional[Callable] = None

        self._initialized = False

    def initialize(self) -> bool:
        """
        GPIO 초기화

        Returns:
            초기화 성공 여부
        """
        if not self.enabled:
            print("GPIO disabled or not available")
            return False

        if self._initialized:
            return True

        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)

            # 입력 핀 설정
            GPIO.setup(self.trigger_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

            # 출력 핀 설정
            GPIO.setup(self.ok_pin, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.ng_pin, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.buzzer_pin, GPIO.OUT, initial=GPIO.LOW)

            self._initialized = True
            print("GPIO initialized successfully")
            return True

        except Exception as e:
            print(f"GPIO initialization failed: {e}")
            self.enabled = False
            return False

    def cleanup(self) -> None:
        """GPIO 정리"""
        if self._initialized and GPIO_AVAILABLE:
            # 모든 출력 OFF
            self.all_off()
            GPIO.cleanup()
            self._initialized = False
            print("GPIO cleaned up")

    def set_trigger_callback(self, callback: Callable) -> None:
        """
        트리거 콜백 함수 설정

        Args:
            callback: 트리거 발생 시 호출할 함수
        """
        if not self.enabled or not self._initialized:
            return

        self._trigger_callback = callback

        # 이벤트 감지 설정 (Rising Edge)
        GPIO.add_event_detect(
            self.trigger_pin,
            GPIO.RISING,
            callback=self._handle_trigger,
            bouncetime=200  # 디바운스 200ms
        )

    def _handle_trigger(self, channel: int) -> None:
        """트리거 이벤트 핸들러"""
        if self._trigger_callback:
            self._trigger_callback()

    def read_trigger(self) -> bool:
        """
        트리거 신호 읽기

        Returns:
            트리거 상태 (True/False)
        """
        if not self.enabled or not self._initialized:
            return False

        return GPIO.input(self.trigger_pin) == GPIO.HIGH

    def wait_for_trigger(self, timeout: float = None) -> bool:
        """
        트리거 대기

        Args:
            timeout: 타임아웃 (초), None이면 무한 대기

        Returns:
            트리거 감지 여부
        """
        if not self.enabled or not self._initialized:
            return False

        if timeout:
            timeout_ms = int(timeout * 1000)
            channel = GPIO.wait_for_edge(
                self.trigger_pin, GPIO.RISING, timeout=timeout_ms
            )
            return channel is not None
        else:
            GPIO.wait_for_edge(self.trigger_pin, GPIO.RISING)
            return True

    def output_ok(self) -> None:
        """OK 신호 출력"""
        if not self.enabled or not self._initialized:
            print("[SIM] OK signal output")
            return

        GPIO.output(self.ok_pin, GPIO.HIGH)
        time.sleep(self.signal_duration)
        GPIO.output(self.ok_pin, GPIO.LOW)

    def output_ng(self, with_buzzer: bool = True) -> None:
        """
        NG 신호 출력

        Args:
            with_buzzer: 버저 울림 여부
        """
        if not self.enabled or not self._initialized:
            print("[SIM] NG signal output")
            return

        GPIO.output(self.ng_pin, GPIO.HIGH)

        if with_buzzer:
            GPIO.output(self.buzzer_pin, GPIO.HIGH)

        time.sleep(self.signal_duration)

        GPIO.output(self.ng_pin, GPIO.LOW)
        GPIO.output(self.buzzer_pin, GPIO.LOW)

    def output_result(self, is_ok: bool) -> None:
        """
        검사 결과 출력

        Args:
            is_ok: 합격 여부
        """
        if is_ok:
            self.output_ok()
        else:
            self.output_ng()

    def buzzer_on(self) -> None:
        """버저 ON"""
        if not self.enabled or not self._initialized:
            return
        GPIO.output(self.buzzer_pin, GPIO.HIGH)

    def buzzer_off(self) -> None:
        """버저 OFF"""
        if not self.enabled or not self._initialized:
            return
        GPIO.output(self.buzzer_pin, GPIO.LOW)

    def buzzer_beep(self, duration: float = 0.1, count: int = 1) -> None:
        """
        버저 비프음

        Args:
            duration: 비프 지속 시간
            count: 비프 횟수
        """
        if not self.enabled or not self._initialized:
            return

        for i in range(count):
            GPIO.output(self.buzzer_pin, GPIO.HIGH)
            time.sleep(duration)
            GPIO.output(self.buzzer_pin, GPIO.LOW)
            if i < count - 1:
                time.sleep(duration)

    def all_off(self) -> None:
        """모든 출력 OFF"""
        if not self.enabled or not self._initialized:
            return

        GPIO.output(self.ok_pin, GPIO.LOW)
        GPIO.output(self.ng_pin, GPIO.LOW)
        GPIO.output(self.buzzer_pin, GPIO.LOW)

    def set_output(self, pin_name: str, state: bool) -> None:
        """
        특정 출력 핀 제어

        Args:
            pin_name: 핀 이름 ('ok', 'ng', 'buzzer')
            state: 상태 (True=HIGH, False=LOW)
        """
        if not self.enabled or not self._initialized:
            return

        pin_map = {
            'ok': self.ok_pin,
            'ng': self.ng_pin,
            'buzzer': self.buzzer_pin
        }

        if pin_name in pin_map:
            GPIO.output(pin_map[pin_name], GPIO.HIGH if state else GPIO.LOW)

    @property
    def is_available(self) -> bool:
        """GPIO 사용 가능 여부"""
        return self.enabled and self._initialized


class MockGPIOController(GPIOController):
    """테스트용 Mock GPIO 컨트롤러"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.enabled = True
        self._states = {
            'trigger': False,
            'ok': False,
            'ng': False,
            'buzzer': False
        }

    def initialize(self) -> bool:
        self._initialized = True
        print("[MOCK] GPIO initialized")
        return True

    def cleanup(self) -> None:
        self._initialized = False
        print("[MOCK] GPIO cleaned up")

    def read_trigger(self) -> bool:
        return self._states['trigger']

    def set_mock_trigger(self, state: bool) -> None:
        """테스트용 트리거 설정"""
        self._states['trigger'] = state
        if state and self._trigger_callback:
            self._trigger_callback()

    def output_ok(self) -> None:
        print("[MOCK] OK signal output")
        self._states['ok'] = True
        time.sleep(0.1)
        self._states['ok'] = False

    def output_ng(self, with_buzzer: bool = True) -> None:
        print("[MOCK] NG signal output")
        self._states['ng'] = True
        if with_buzzer:
            self._states['buzzer'] = True
        time.sleep(0.1)
        self._states['ng'] = False
        self._states['buzzer'] = False
