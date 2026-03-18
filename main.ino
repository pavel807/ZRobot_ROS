#include <NewPing.h>

#define TRIGGER_PIN  11
#define ECHO_PIN     12
#define MAX_DISTANCE 400
NewPing sonar(TRIGGER_PIN, ECHO_PIN, MAX_DISTANCE);

// Пины двигателей
int enA = 9; int in1 = 7; int in2 = 6;
int enB = 3; int in3 = 4; int in4 = 5;

const int sensorPin = 14;
const int sensorPinTwo = 10;

// Целевые скорости от RK3588
int targetSpeedL = 0; 
int targetSpeedR = 0;
// Фактические скорости на моторах
int actualSpeedL = 0;
int actualSpeedR = 0;

// Флаг состояния
bool emergencyStop = false;
unsigned long lastCmdTime = 0;
const unsigned long CMD_TIMEOUT = 1500; // Таймаут команд 1500мс (под web/UI без постоянного стрима)

void setup() {
  Serial.begin(9600);
  
  pinMode(enA, OUTPUT); 
  pinMode(enB, OUTPUT);
  pinMode(in1, OUTPUT); 
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT); 
  pinMode(in4, OUTPUT);
  pinMode(sensorPin, INPUT);
  pinMode(sensorPinTwo, INPUT);
  
  stopMotors();
  
  Serial.println("System ready. Waiting commands...");
  Serial.println("Format: /motor L 255");
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

void setMotorSpeed(int motor, int speed, int inPin1, int inPin2, int enPin) {
  speed = constrain(speed, 0, 255);
  
  if (speed == 0) {
    digitalWrite(inPin1, LOW);
    digitalWrite(inPin2, LOW);
  } else {
    digitalWrite(inPin1, LOW);
    digitalWrite(inPin2, HIGH);
  }
  analogWrite(enPin, speed);
  
  if (motor == 0) actualSpeedL = speed;
  else actualSpeedR = speed;
}

void applySpeeds() {
  setMotorSpeed(0, targetSpeedL, in1, in2, enA);
  setMotorSpeed(1, targetSpeedR, in3, in4, enB);
}

void loop() {
  // 1. Чтение команд с таймаутом
  while (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    lastCmdTime = millis();

    if (input.startsWith("/motor")) {
      char side = input.charAt(7); 
      int speed = input.substring(9).toInt();
      speed = constrain(speed, 0, 255);

      if (side == 'L' || side == 'l') {
        targetSpeedL = speed;
        Serial.print("L:"); Serial.println(targetSpeedL);
      } else if (side == 'R' || side == 'r') {
        targetSpeedR = speed;
        Serial.print("R:"); Serial.println(targetSpeedR);
      }
    }
  }

  // Проверка таймаута команд - остановка если долго нет связи
  if (millis() - lastCmdTime > CMD_TIMEOUT) {
    if (targetSpeedL != 0 || targetSpeedR != 0) {
      targetSpeedL = 0;
      targetSpeedR = 0;
      Serial.println("TIMEOUT - STOP");
    }
  }

  // 2. Чтение датчиков
  int distance = sonar.ping_cm();
  int irLeft = digitalRead(sensorPin);
  int irRight = digitalRead(sensorPinTwo);

  // 3. Логика безопасности
  bool pathClear = (irLeft == LOW && irRight == LOW && (distance > 40 || distance == 0));
  
  if (!pathClear) {
    if (!emergencyStop) {
      stopMotors();
      emergencyStop = true;
      Serial.print("EMERGENCY! D:"); 
      Serial.print(distance);
      Serial.print(" IR:"); 
      Serial.print(irLeft);
      Serial.print("/"); 
      Serial.println(irRight);
    }
  } else {
    emergencyStop = false;
    
    // Применяем целевые скорости только если они изменились
    if (targetSpeedL != actualSpeedL || targetSpeedR != actualSpeedR) {
      applySpeeds();
    }
  }

  // Отладка
  static unsigned long lastDebug = 0;
  if (millis() - lastDebug > 500) {
    Serial.print("T:"); Serial.print(targetSpeedL);
    Serial.print("/"); Serial.print(targetSpeedR);
    Serial.print(" A:"); Serial.print(actualSpeedL);
    Serial.print("/"); Serial.print(actualSpeedR);
    Serial.print(" D:"); Serial.println(distance);
    lastDebug = millis();
  }
  
  delay(5);
}