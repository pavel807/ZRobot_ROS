#include <NewPing.h>

// Настройки ультразвука
#define TRIGGER_PIN  11
#define ECHO_PIN     12
#define MAX_DISTANCE 400
NewPing sonar(TRIGGER_PIN, ECHO_PIN, MAX_DISTANCE);

// Пины двигателей L298N
const int enA = 9; 
const int in1 = 7; 
const int in2 = 6;
const int enB = 3; 
const int in3 = 4; 
const int in4 = 5;

// Состояние скоростей
int targetSpeedL = 0; 
int targetSpeedR = 0;
int actualSpeedL = 0;
int actualSpeedR = 0;

// Флаги и таймеры
bool emergencyStop = false;
unsigned long lastCmdTime = 0;
const unsigned long CMD_TIMEOUT = 1000; // Увеличил до 1с для стабильности при старте
unsigned long lastDebug = 0;
unsigned long lastPing = 0;

void setup() {
  // 1. Инициализация порта
  Serial.begin(115200);

  // 2. Ожидание подключения (Критично для того, чтобы компьютер успел увидеть порт)
  while (!Serial) {
    delay(10); 
  }

  // 3. Пауза вежливости (даем ROS2 время прочухаться после открытия порта)
  delay(2000); 
  
  // Настройка пинов
  pinMode(enA, OUTPUT); 
  pinMode(enB, OUTPUT);
  pinMode(in1, OUTPUT); 
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT); 
  pinMode(in4, OUTPUT);
  
  // Начальное состояние — стоп
  stopMotors();
  
  // Сбрасываем таймер команд на текущее время, чтобы не сработал таймаут сразу
  lastCmdTime = millis();
  
  Serial.println("SYSTEM:READY");
  Serial.println("PING:OK");
}

void stopMotors() {
  digitalWrite(in1, LOW); 
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW); 
  digitalWrite(in4, LOW);
  analogWrite(enA, 0);
  analogWrite(enB, 0);
  actualSpeedL = 0;
  actualSpeedR = 0;
}

void setMotorSpeed(int motor, int speed, int p1, int p2, int en) {
  speed = constrain(speed, 0, 255);
  if (speed == 0) {
    digitalWrite(p1, LOW);
    digitalWrite(p2, LOW);
  } else {
    digitalWrite(p1, LOW);
    digitalWrite(p2, HIGH);
  }
  analogWrite(en, speed);

  if (motor == 0) actualSpeedL = speed;
  else actualSpeedR = speed;
}

void applySpeeds() {
  setMotorSpeed(0, targetSpeedL, in1, in2, enA);
  setMotorSpeed(1, targetSpeedR, in3, in4, enB);
}

void loop() {
  // --- 1. ЧТЕНИЕ КОМАНД ---
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input.startsWith("/motor")) {
      lastCmdTime = millis(); // Обновляем время жизни связи
      
      // Парсинг: /motor L 150 или /motor R 150
      if (input.length() > 8) {
        char side = input.charAt(7); 
        int spaceIndex = input.indexOf(' ', 8);
        if (spaceIndex != -1) {
          int speed = input.substring(spaceIndex + 1).toInt();
          speed = constrain(speed, 0, 255);

          if (side == 'L' || side == 'l') {
            targetSpeedL = speed;
          } else if (side == 'R' || side == 'r') {
            targetSpeedR = speed;
          }
        }
      }
    }
  }

  // --- 2. ПРОВЕРКА ТАЙМАУТА (Safety) ---
  if (millis() - lastCmdTime > CMD_TIMEOUT) {
    if (targetSpeedL != 0 || targetSpeedR != 0) {
      targetSpeedL = 0;
      targetSpeedR = 0;
      stopMotors();
      Serial.println("ERROR:TIMEOUT_STOP");
    }
  }

  // --- 3. ДАТЧИКИ И БЕЗОПАСНОСТЬ ---
  int distance = sonar.ping_cm();
  // Если расстояние 0, значит препятствий нет (датчик NewPing возвращает 0 при выходе за предел)
  bool pathClear = (distance > 30 || distance == 0); 
  
  if (!pathClear) {
    if (!emergencyStop) {
      stopMotors();
      emergencyStop = true;
      Serial.print("ALARM:OBSTACLE_AT_"); 
      Serial.println(distance);
    }
  } else {
    emergencyStop = false;
    // Применяем скорость только если путь чист
    if (targetSpeedL != actualSpeedL || targetSpeedR != actualSpeedR) {
      applySpeeds();
    }
  }

  // --- 4. ОБРАТНАЯ СВЯЗЬ (Телеметрия) ---
  unsigned long now = millis();
  
  // Каждую секунду подтверждаем связь
  if (now - lastPing > 1000) {
    Serial.println("PING:OK");
    lastPing = now;
  }

  // Каждые 500мс шлем состояние в ROS
  if (now - lastDebug > 500) {
    Serial.print("DATA:"); 
    Serial.print(actualSpeedL); Serial.print(",");
    Serial.print(actualSpeedR); Serial.print(",");
    Serial.println(distance);
    lastDebug = now;
  }
  
  delay(10); // Маленькая пауза для стабильности цикла
}
