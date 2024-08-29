#include <IBusBM.h>

IBusBM IBus; 

///driver
int motorRightPin1 = 12;
int motorRightPin2 = 13;

int motorLeftPin1 = 18;
int motorLeftPin2 = 19;

/// pompa 1
int relay1Pin = 21;
/// pompa 2
int relay2Pin = 22;
/// fan dc
int relay3Pin = 23;
// kirim sinyal 0/1 ke jetson
int onoff =14;
/// indikator mode manual
int ledKuning = 32;
/// indikator mode otomatis
int ledHijau = 33;

///durasi belok
int turnDuration = 3500;

int motorSpeed;
int motorTurn;

unsigned long lastCommandTime = 0;
unsigned long lastReceiveTime = 0; // variabel untuk waktu terakhir menerima data

unsigned long previousTime = 0;
unsigned long sampleTime = 100; // Interval waktu untuk perhitungan PID 

const unsigned long timeoutDuration = 100;
const unsigned long timeoutDurationSTOP = 500;
const unsigned long timeoutDurationError = 500;
const unsigned long stopDuration = 1000; // Durasi untuk berhenti setelah tidak menerima data


String receivedData;
float errorValue;

float previousError = 0;
float integral = 0;
float pidValue = 0;

float kp = 3.5;
float ki = 0.00005;
float kd = 0.4;

///channel FS
int ch_3, ch_1, ch_5, ch_6, ch_7, ch_8;
  
void setup() {
  Serial.begin(115200);
  IBus.begin(Serial2,1);

  pinMode(relay1Pin, OUTPUT);
  pinMode(relay2Pin, OUTPUT);
  pinMode(relay3Pin, OUTPUT);
  pinMode(onoff, OUTPUT);
  pinMode(ledKuning, OUTPUT);
  pinMode(ledHijau, OUTPUT);
  pinMode(motorRightPin1, OUTPUT);
  pinMode(motorRightPin2, OUTPUT);
  pinMode(motorLeftPin1, OUTPUT);
  pinMode(motorLeftPin2, OUTPUT);
  digitalWrite(relay1Pin, LOW);
  digitalWrite(relay2Pin, LOW);
  digitalWrite(relay3Pin, LOW);
  digitalWrite(onoff, LOW);
  digitalWrite(ledKuning, LOW);
  digitalWrite(ledHijau, LOW);
  analogWrite(motorRightPin1, 0);
  analogWrite(motorRightPin2, 0);
  analogWrite(motorLeftPin1, 0);
  analogWrite(motorLeftPin2, 0);
}

void loop() {
  
  /// maju mundur
  ch_3 = IBus.readChannel(2);
  
  /// belok kanan kiri
  ch_1 = IBus.readChannel(0);
  
  /// auto manual
  ch_5 = IBus.readChannel(4);
  
  /// fan DC
  ch_6 = IBus.readChannel(5);
  
  /// pompa 1
  ch_7 = IBus.readChannel(6);
  
  /// pompa 2
  ch_8 = IBus.readChannel(7);

  motorSpeed = map(ch_3, 1000, 2000, -80, 80);
  motorTurn = map(ch_1, 1000, 2000, -80, 80);

  if (ch_5 > 1500) {
    digitalWrite(onoff, HIGH);
    digitalWrite(ledKuning, LOW);
    digitalWrite(ledHijau, HIGH);
    ch_1 = 1500;
    ch_3 = 1500;
    ch_5 = 1500;
    ch_6 = 1500;
    ch_7 = 1500;
    ch_8 = 1500;
    otomatis();
  }

  else {
    digitalWrite(ledKuning, HIGH);
    digitalWrite(ledHijau, LOW);
    digitalWrite(onoff, LOW); 
    if (motorTurn > 5){
      kanan(motorTurn, motorTurn);
  //    kananN(motorTurn, motorTurn);/
    } else if (motorTurn < -5){
      kiri(motorTurn, motorTurn);
  //    kiriI(motorTurn, motorTurn);/
    } else if (motorSpeed < 5 && motorSpeed > -5 && motorTurn > -5 && motorTurn < 5){
      stopp();
    }
      else if (motorSpeed > 5 && motorTurn > -5 && motorTurn < 5){
      maju(motorSpeed, motorSpeed);
    } else if (motorSpeed < 5 && motorTurn > -5 && motorTurn < 5){
      mundur(motorSpeed, motorSpeed);
    }

    
    if (ch_7 > 1500) {
      digitalWrite(relay1Pin, HIGH);
    } else {
      digitalWrite(relay1Pin, LOW);
    }
  
    if (ch_8 > 1500) {
      digitalWrite(relay2Pin, HIGH);
    } else {
      digitalWrite(relay2Pin, LOW);
    }
  
    if (ch_6 > 1500) {
      digitalWrite(relay3Pin, HIGH);
    } else {
      digitalWrite(relay3Pin, LOW);
    }

  }

  delay(100);

}

void maju(int leftSpeed, int rightSpeed) {
  analogWrite(motorLeftPin1, abs(leftSpeed));
  analogWrite(motorLeftPin2, 0);
  analogWrite(motorRightPin1, abs(rightSpeed));
  analogWrite(motorRightPin2, 0);
}

void mundur(int leftSpeed, int rightSpeed) {
  analogWrite(motorLeftPin1, 0);
  analogWrite(motorLeftPin2, abs(leftSpeed));
  analogWrite(motorRightPin1, 0);
  analogWrite(motorRightPin2, abs(rightSpeed));
}

void kanan(int leftSpeed, int rightSpeed) {
  analogWrite(motorLeftPin1, abs(leftSpeed));
  analogWrite(motorLeftPin2, 0);
  analogWrite(motorRightPin1, 0);
  analogWrite(motorRightPin2, abs(rightSpeed));
}

void kiri(int leftSpeed, int rightSpeed) {
  analogWrite(motorLeftPin1, 0);
  analogWrite(motorLeftPin2, abs(leftSpeed));
  analogWrite(motorRightPin1, abs(rightSpeed));
  analogWrite(motorRightPin2, 0);
}

void majuU() {
  analogWrite(motorLeftPin1, 45);
  analogWrite(motorLeftPin2, 0);
  analogWrite(motorRightPin1, 45);
  analogWrite(motorRightPin2, 0);
}

void kananUJ() {
  analogWrite(motorLeftPin1, 70);
  analogWrite(motorLeftPin2, 0);
  analogWrite(motorRightPin1, 0);
  analogWrite(motorRightPin2, 32);
}

void kiriUJ() {
  analogWrite(motorLeftPin1, 0);
  analogWrite(motorLeftPin2, 32);
  analogWrite(motorRightPin1, 70);
  analogWrite(motorRightPin2, 0);
}

void stopp() {
  analogWrite(motorLeftPin1, 0);
  analogWrite(motorLeftPin2, 0);
  analogWrite(motorRightPin1, 0);
  analogWrite(motorRightPin2, 0);
}

void otomatis() {
  unsigned long currentTime = millis();
  unsigned long elapsedTime = currentTime - previousTime;

  // Pastikan waktu yang telah berlalu mencapai sampleTime sebelum melakukan perhitungan PID
  if (elapsedTime >= sampleTime) {
    // Reset previousTime ke currentTime
    previousTime = currentTime;

    if (Serial.available() > 0) {
      receivedData = Serial.readStringUntil('\n');
      Serial.print("receivedData: ");
      Serial.println(receivedData);
      lastReceiveTime = millis();

      if (receivedData.startsWith("error_")) {
        errorValue = receivedData.substring(6).toFloat();
//        Serial.print("errorValue: ");
//        Serial.println(errorValue);
        if (currentTime - lastCommandTime > timeoutDuration) {
          float derivative = errorValue - previousError;
          pidValue = calculatePID(errorValue, integral, derivative);
          previousError = errorValue;
          integral += errorValue;

          adjustMotorSpeeds(pidValue);
//          lastCommandTime = millis();
          lastCommandTime = currentTime;
        }
      } 
      
      else if (receivedData == "KI") {
//        if (millis() - lastCommandTime > timeoutDurationKI) {
        Serial.println("KIRI BWANG");
        Serial.println("");
//        majuU();
//        delay(3000);
        kiriUJ();
        delay(turnDuration);
//        stopp();
//        delay(500);
//          lastCommandTime = currentTime;
        lastCommandTime = millis();
//        }
      } 
      
      else if (receivedData == "KA") {
//        if (millis() - lastCommandTime > timeoutDurationKA) {
        Serial.println("KANAN BWANG");
        Serial.println("");
//        majuU();
//        delay(3000);
        kananUJ();
        delay(turnDuration);
//        stopp();
//        delay(500);
//          lastCommandTime = currentTime;
        lastCommandTime = millis();          
//        }
      } 

      else if (receivedData == "MD") {
        if (millis() - lastCommandTime) {
          Serial.println("MAJU DIKIT BWANG");
          Serial.println("");
          majuU();
//          stopp();
//          delay(1000);
//          lastCommandTime = currentTime;
          lastCommandTime = millis();
        }
      }
      
      else {
        handleSerialInput(receivedData);
      }
    }

    // Stop motor jika tidak ada data yang diterima lagi
    if (millis() - lastReceiveTime > stopDuration && lastReceiveTime != 0) {
      stopp();
      delay(100);
    }
    
  }
}

void handleSerialInput(String input) {
  if (input.startsWith("p=")) {
    kp = input.substring(2).toFloat();
//    Serial.print("Kp diatur ke: ");
//    Serial.println(kp);
  } else if (input.startsWith("i=")) {
    ki = input.substring(2).toFloat();
//    Serial.print("Ki diatur ke: ");
//    Serial.println(ki);
  } else if (input.startsWith("d=")) {
    kd = input.substring(2).toFloat();
//    Serial.print("Kd diatur ke: ");
//    Serial.println(kd);
  }
}

float calculatePID(float errorValue, float integral, float derivative) {
  float pidValue = kp * errorValue + ki * integral + kd * derivative;
  Serial.print("kp: ");
  Serial.println(kp, 6);
  Serial.print("ki: ");
  Serial.println(ki, 6);
  Serial.print("kd: ");
  Serial.println(kd, 6);
//  Serial.print("errorValue: ");
//  Serial.println(errorValue);
//  Serial.print("pidValue: ");
//  Serial.println(pidValue, 6);
//  Serial.println(" ");
  return pidValue;
}

// PID UNTUK LURUS
void adjustMotorSpeeds(float pidValue) {
//  int maxMotorSpeed = 50; // Maximum speed value
//  int minMotorSpeed = -50; // Minimum speed value (for reverse)
  int maxMotorSpeed = 65;// Maximum speed value
  int minMotorSpeed = -65; // Minimum speed value (for reverse)

  // Calculate base speeds
//  int baseSpeed = 40; // Base speed for both motors
  int baseSpeed = 55;// Base speed for both motors
  
  // Adjust speeds based on PID value
  int rightSpeed = baseSpeed - pidValue;
  int leftSpeed = baseSpeed + pidValue;

  // Constrain speeds to maximum and minimum limits
  leftSpeed = constrain(leftSpeed, minMotorSpeed, maxMotorSpeed);
  rightSpeed = constrain(rightSpeed, minMotorSpeed, maxMotorSpeed);

  // Control left motor
  if (leftSpeed >= 0) {
    analogWrite(motorLeftPin1, abs(leftSpeed));
    analogWrite(motorLeftPin2, 0);
  } else {
    analogWrite(motorLeftPin1, 0);
    analogWrite(motorLeftPin2, abs(leftSpeed)); // Use abs() to ensure positive value
  }

  // Control right motor
  if (rightSpeed >= 0) {
    analogWrite(motorRightPin1, abs(rightSpeed));
    analogWrite(motorRightPin2, 0);
  } else {
    analogWrite(motorRightPin1, 0);
    analogWrite(motorRightPin2, abs(rightSpeed)); // Use abs() to ensure positive value
  }

//  Serial.print("Time: ");
//  Serial.print(millis());
  Serial.print("Error: ");
//  Serial.print(",");
  Serial.print(errorValue);
  Serial.print(", PID: ");
//  Serial.print(",");
  Serial.print(pidValue);
  Serial.print(", Left Speed: ");
//  Serial.print(",");
  Serial.print(leftSpeed);
  Serial.print(", Right Speed: ");
//  Serial.print(",");
  Serial.println(rightSpeed);
}

void manual() {
  if (ch_5 < 1500) {
    stopp();
  }
}
