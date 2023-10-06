# Executable File make

- use pyinstaller

```
  pyinstaller -n "Road Classification" --add-data best_model.h5;model main.py
```

# Main Structure
- main.py
  GUI 구현 내용
- utils/etc/observe.py
  파일 생성 이벤트 감지 및 실시간 알고리즘이 동작하는 쓰레드의 시작, 종료 관리
- utils/send_receive/mqtt_sender.py
  observe.py에서 감지된 파일에 따라 특정 알고리즘을 반복 수행하는 쓰레드