"""
데이터베이스 관리 모듈
Database Management Module for Inspection Results

지원 데이터베이스:
- SQLite (기본)
- MySQL (선택)
- PostgreSQL (선택)
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import asdict


class DatabaseManager:
    """데이터베이스 관리 클래스"""

    def __init__(self, config: dict):
        """
        Args:
            config: 데이터베이스 설정 딕셔너리
                - enabled: 사용 여부
                - type: 데이터베이스 종류 (sqlite, mysql, postgresql)
                - path: SQLite 파일 경로
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.db_type = config.get('type', 'sqlite')
        self.db_path = config.get('path', 'logs/inspection.db')

        self.connection = None
        self._initialized = False

    def initialize(self) -> bool:
        """
        데이터베이스 초기화

        Returns:
            초기화 성공 여부
        """
        if not self.enabled:
            print("Database disabled")
            return False

        try:
            if self.db_type == 'sqlite':
                return self._init_sqlite()
            else:
                print(f"Database type '{self.db_type}' not supported yet")
                return False

        except Exception as e:
            print(f"Database initialization failed: {e}")
            return False

    def _init_sqlite(self) -> bool:
        """SQLite 초기화"""
        # 디렉토리 생성
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row

        # 테이블 생성
        self._create_tables()

        self._initialized = True
        print(f"SQLite database initialized: {self.db_path}")
        return True

    def _create_tables(self) -> None:
        """테이블 생성"""
        cursor = self.connection.cursor()

        # 검사 결과 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inspection_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                part_id TEXT,
                barcode TEXT,
                defect_ok INTEGER,
                barcode_ok INTEGER,
                variant_ok INTEGER,
                final_result TEXT,
                ng_reason TEXT,
                inspection_time_ms REAL,
                details TEXT,
                image_path TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 결함 상세 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS defects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id INTEGER,
                defect_type TEXT,
                bbox TEXT,
                area REAL,
                severity TEXT,
                confidence REAL,
                FOREIGN KEY (inspection_id) REFERENCES inspection_results(id)
            )
        ''')

        # 통계 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE,
                total_count INTEGER DEFAULT 0,
                ok_count INTEGER DEFAULT 0,
                ng_count INTEGER DEFAULT 0,
                defect_ng INTEGER DEFAULT 0,
                barcode_ng INTEGER DEFAULT 0,
                variant_ng INTEGER DEFAULT 0,
                updated_at TEXT
            )
        ''')

        # 인덱스 생성
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON inspection_results(timestamp)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_final_result
            ON inspection_results(final_result)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_part_id
            ON inspection_results(part_id)
        ''')

        self.connection.commit()

    def save_result(self, result: Any, image_path: str = None) -> int:
        """
        검사 결과 저장

        Args:
            result: InspectionResult 객체
            image_path: 이미지 저장 경로

        Returns:
            저장된 레코드 ID
        """
        if not self._initialized:
            return -1

        cursor = self.connection.cursor()

        # 결과를 딕셔너리로 변환
        if hasattr(result, '__dict__'):
            result_dict = asdict(result) if hasattr(result, '__dataclass_fields__') else result.__dict__
        else:
            result_dict = result

        # 메인 결과 저장
        cursor.execute('''
            INSERT INTO inspection_results (
                timestamp, part_id, barcode,
                defect_ok, barcode_ok, variant_ok,
                final_result, ng_reason, inspection_time_ms,
                details, image_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result_dict.get('timestamp', datetime.now().isoformat()),
            result_dict.get('part_id'),
            result_dict.get('barcode'),
            1 if result_dict.get('defect_ok') else 0,
            1 if result_dict.get('barcode_ok') else 0,
            1 if result_dict.get('variant_ok') else 0,
            result_dict.get('final_result'),
            result_dict.get('ng_reason'),
            result_dict.get('inspection_time_ms'),
            json.dumps(result_dict, ensure_ascii=False),
            image_path
        ))

        inspection_id = cursor.lastrowid

        # 결함 상세 저장
        defects = result_dict.get('defects', [])
        for defect in defects:
            cursor.execute('''
                INSERT INTO defects (
                    inspection_id, defect_type, bbox, area, severity, confidence
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                inspection_id,
                defect.get('type'),
                json.dumps(defect.get('bbox')),
                defect.get('area'),
                defect.get('severity'),
                defect.get('confidence')
            ))

        # 일일 통계 업데이트
        self._update_daily_stats(result_dict)

        self.connection.commit()
        return inspection_id

    def _update_daily_stats(self, result: dict) -> None:
        """일일 통계 업데이트"""
        cursor = self.connection.cursor()
        today = datetime.now().strftime('%Y-%m-%d')

        # 기존 통계 조회
        cursor.execute(
            'SELECT * FROM daily_stats WHERE date = ?', (today,)
        )
        row = cursor.fetchone()

        if row:
            # 업데이트
            total = row['total_count'] + 1
            ok = row['ok_count'] + (1 if result.get('final_result') == 'OK' else 0)
            ng = row['ng_count'] + (1 if result.get('final_result') == 'NG' else 0)
            defect_ng = row['defect_ng'] + (0 if result.get('defect_ok') else 1)
            barcode_ng = row['barcode_ng'] + (0 if result.get('barcode_ok') else 1)
            variant_ng = row['variant_ng'] + (0 if result.get('variant_ok') else 1)

            cursor.execute('''
                UPDATE daily_stats SET
                    total_count = ?, ok_count = ?, ng_count = ?,
                    defect_ng = ?, barcode_ng = ?, variant_ng = ?,
                    updated_at = ?
                WHERE date = ?
            ''', (total, ok, ng, defect_ng, barcode_ng, variant_ng,
                  datetime.now().isoformat(), today))
        else:
            # 신규 생성
            cursor.execute('''
                INSERT INTO daily_stats (
                    date, total_count, ok_count, ng_count,
                    defect_ng, barcode_ng, variant_ng, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                today,
                1,
                1 if result.get('final_result') == 'OK' else 0,
                1 if result.get('final_result') == 'NG' else 0,
                0 if result.get('defect_ok') else 1,
                0 if result.get('barcode_ok') else 1,
                0 if result.get('variant_ok') else 1,
                datetime.now().isoformat()
            ))

    def get_results(self, limit: int = 100,
                    start_date: str = None,
                    end_date: str = None,
                    result_filter: str = None) -> List[Dict]:
        """
        검사 결과 조회

        Args:
            limit: 최대 조회 수
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
            result_filter: 결과 필터 ('OK', 'NG')

        Returns:
            검사 결과 리스트
        """
        if not self._initialized:
            return []

        cursor = self.connection.cursor()

        query = 'SELECT * FROM inspection_results WHERE 1=1'
        params = []

        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date)

        if end_date:
            query += ' AND timestamp <= ?'
            params.append(end_date + 'T23:59:59')

        if result_filter:
            query += ' AND final_result = ?'
            params.append(result_filter)

        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_daily_stats(self, date: str = None) -> Optional[Dict]:
        """
        일일 통계 조회

        Args:
            date: 날짜 (YYYY-MM-DD), None이면 오늘

        Returns:
            통계 딕셔너리
        """
        if not self._initialized:
            return None

        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        cursor = self.connection.cursor()
        cursor.execute('SELECT * FROM daily_stats WHERE date = ?', (date,))
        row = cursor.fetchone()

        if row:
            stats = dict(row)
            # 비율 계산
            total = stats['total_count']
            if total > 0:
                stats['ok_rate'] = stats['ok_count'] / total * 100
                stats['ng_rate'] = stats['ng_count'] / total * 100
            else:
                stats['ok_rate'] = 0
                stats['ng_rate'] = 0
            return stats

        return None

    def get_stats_range(self, days: int = 7) -> List[Dict]:
        """
        기간별 통계 조회

        Args:
            days: 조회 기간 (일)

        Returns:
            일별 통계 리스트
        """
        if not self._initialized:
            return []

        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        cursor = self.connection.cursor()
        cursor.execute('''
            SELECT * FROM daily_stats
            WHERE date >= ?
            ORDER BY date DESC
        ''', (start_date,))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_ng_analysis(self, days: int = 30) -> Dict:
        """
        NG 원인 분석

        Args:
            days: 분석 기간 (일)

        Returns:
            NG 원인별 통계
        """
        if not self._initialized:
            return {}

        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        cursor = self.connection.cursor()

        # NG 사유별 집계
        cursor.execute('''
            SELECT ng_reason, COUNT(*) as count
            FROM inspection_results
            WHERE final_result = 'NG' AND timestamp >= ?
            GROUP BY ng_reason
            ORDER BY count DESC
        ''', (start_date,))

        reasons = {row['ng_reason']: row['count'] for row in cursor.fetchall()}

        # 결함 유형별 집계
        cursor.execute('''
            SELECT defect_type, COUNT(*) as count
            FROM defects d
            JOIN inspection_results r ON d.inspection_id = r.id
            WHERE r.timestamp >= ?
            GROUP BY defect_type
            ORDER BY count DESC
        ''', (start_date,))

        defect_types = {row['defect_type']: row['count'] for row in cursor.fetchall()}

        return {
            'ng_reasons': reasons,
            'defect_types': defect_types
        }

    def cleanup_old_records(self, days: int = 30) -> int:
        """
        오래된 레코드 삭제

        Args:
            days: 보관 기간 (일)

        Returns:
            삭제된 레코드 수
        """
        if not self._initialized:
            return 0

        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        cursor = self.connection.cursor()

        # 관련 결함 삭제
        cursor.execute('''
            DELETE FROM defects WHERE inspection_id IN (
                SELECT id FROM inspection_results WHERE timestamp < ?
            )
        ''', (cutoff_date,))

        # 검사 결과 삭제
        cursor.execute(
            'DELETE FROM inspection_results WHERE timestamp < ?',
            (cutoff_date,)
        )

        deleted = cursor.rowcount
        self.connection.commit()

        print(f"Deleted {deleted} old records")
        return deleted

    def close(self) -> None:
        """데이터베이스 연결 종료"""
        if self.connection:
            self.connection.close()
            self.connection = None
            self._initialized = False
            print("Database connection closed")

    @property
    def is_available(self) -> bool:
        """데이터베이스 사용 가능 여부"""
        return self._initialized
