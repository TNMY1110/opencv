import serial
import time

# --- stub 함수: 아직 구현하지 않음 ---
def send_command(ser, command):
    """아두이노에 명령 전송 (GREEN에서 구현할 예정)"""
    return False   # 지금은 항상 False 반환

# --- 기대값 확인 ---
# 참고: ser=None으로 설정하여 아두이노가 없어도 테스트 가능
ser = None

# send_command()가 아직 구현되지 않았으므로 False 반환
result = send_command(ser, 'O')

# RED 단계: 이 부분이 실행되어야 함 (즉, FAIL이 출력되어야 함)
if result:
    print("✅ PASS: 아두이노 명령 전송 성공!")
else:
    print("❌ FAIL: send_command() 함수가 아직 구현되지 않았습니다")

# 추가로 'C' 명령도 테스트
result = send_command(ser, 'C')

if result:
    print("✅ PASS: 문 닫기 명령 전송 성공!")
else:
    print("❌ FAIL: send_command() 함수가 아직 구현되지 않았습니다")