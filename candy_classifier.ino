#include <Camera.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <Arduino.h>
#include "model.h"
#include "allergen_database.h"
#include "string.h"

#include <Servo.h>

static Servo myservo1; 
static Servo myservo2;// create servo object to control a servo
// twelve servo objects can be created on most boards

tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 500000;
uint8_t tensor_arena[kTensorArenaSize];


const int offset_x = 32;
const int offset_y = 12;
const int width    = 160;
const int height   = 120;
const int target_w = 96;
const int target_h = 96;
const int pixfmt   = CAM_IMAGE_PIX_FMT_YUV422;

void CamCB(CamImage img) {
  static uint32_t last_mills = 0;
  int8_t max_score = 0;

  if (!img.isAvailable()) {
    Serial.println("img is not available");
    return;
  }

  uint16_t* buf = (uint16_t*)img.getImgBuff();   
  int n = 0; 
  for (int y = offset_y; y < offset_y + target_h; ++y) {
    for (int x = offset_x; x < offset_x + target_w; ++x) {
      uint16_t value = buf[y*width + x];
      uint16_t y_h = (value & 0xf000) >> 8;
      uint16_t y_l = (value & 0x00f0) >> 4;
      value = (y_h | y_l);      
      input->data.f[n++] = (float)(value)/255.0;
    }
  }
  static int count = 0;
  static int calibration_flag = 0;
  static int averaging_setup_skittles[20] = {0};
  static int averaging_setup_twix[20] = {0};
  static int averaging_setup_snickers[20] = {0};
  
  static int sum_snickers = 0;
  static int sum_twix = 0;
  static int sum_skittles = 0;
  
  static int min_snickers = 0;
  static int min_skittles = 0;
  static int min_twix = 0;
  
  static int max_snickers = 0;
  static int max_skittles = 0;
  static int max_twix = 0;

  static int delta_skittles = 0;
  static int delta_snickers = 0;
  static int delta_twix = 0;

  static int abs_delta_skittles = 0;
  static int abs_delta_snickers = 0;
  static int abs_delta_twix = 0;
  
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }
  uint32_t current_mills = millis();
  uint32_t duration = current_mills - last_mills;
  Serial.println("duration = " + String(duration));
  last_mills = current_mills; 


  
  bool result = false;
  int8_t candy_score0 = int8_t(output->data.int8[0])/2;
  int8_t candy_score1 = output->data.int8[1];
  int8_t candy_score2 = output->data.int8[2];
  int8_t candy_score3 = output->data.int8[3];

  if (count < 20)
  {
    averaging_setup_snickers[count] = candy_score0;
    averaging_setup_skittles[count] = candy_score1;
    averaging_setup_twix[count] = candy_score2;
    count+= 1;
  }
  else if (count == 20)
  {
    calibration_flag = 1;
    for (int i = 0; i < 20; i++) 
    {
      sum_snickers += averaging_setup_snickers[i];
      sum_skittles += averaging_setup_skittles[i];
      sum_twix += averaging_setup_twix[i];
    }

//    min_snickers = min(min_snickers, averaging_setup_snickers[i]);
//    min_skittles = min(min_skittles, averaging_setup_skittles);
//    min_twix = min(min_twix, averaging_setup_twix);
    for (int i = 0; i < 20; i++) 
    {
    max_snickers = max(max_snickers, averaging_setup_snickers[i]);
    max_skittles = max(max_skittles, averaging_setup_skittles[i]);
    max_twix =     max(max_twix,      averaging_setup_twix[i]);
    }
    
    sum_snickers = sum_snickers / 20.0;
    sum_skittles = sum_skittles / 20.0;
    sum_twix = sum_twix / 20.0;
  }

  if (calibration_flag)
  {
    Serial.println("--------------------------------");
      delta_snickers = (sum_snickers - candy_score0);
      delta_skittles = (sum_skittles - candy_score1);
      delta_twix = (sum_twix - candy_score2);

      abs_delta_snickers = abs(sum_snickers - candy_score0);
      abs_delta_skittles = abs(sum_skittles - candy_score1);
      abs_delta_twix = abs(sum_twix - candy_score2);

//    candy_score0 = candy_score0 - max_snickers;
//    candy_score1 = candy_score1 - max_skittles;
//    candy_score2 = candy_score2 - max_twix; 
//  
  
    max_score = (delta_snickers > delta_skittles)?delta_snickers:delta_skittles;
    max_score = (max_score > delta_twix)?max_score:delta_twix;
    
    Serial.println("Snickers " + String(candy_score0) + " delt " + delta_snickers + " abs " + abs_delta_snickers);
    Serial.println("Skittles " + String(candy_score1) + " delt " + delta_skittles + " abs " + abs_delta_skittles);
    Serial.println("Twix     " + String(candy_score2) + " delt " + delta_twix +     " abs " + abs_delta_twix);
    Serial.println(String(delta_twix+delta_skittles+delta_snickers));
    Serial.println(String(abs_delta_twix+abs_delta_skittles+abs_delta_snickers));

    const char* candy = ((abs_delta_twix+abs_delta_skittles+abs_delta_snickers)<100)?"Nothing":(max_score == delta_snickers)? 
    "Snickers": ((max_score == delta_twix)? "Twix":(max_score == delta_skittles)? "Skittles": "Nothing??");
    
    if (delta_twix>2 && abs(delta_skittles)<100)
    {
      candy = "Twix";
    }

    Serial.print("Its ");
    Serial.println(candy);

    max_score = (abs_delta_snickers > abs_delta_skittles)?abs_delta_snickers:abs_delta_skittles;
    max_score = (max_score > abs_delta_twix)?max_score:abs_delta_twix;
    candy = ((abs_delta_snickers+abs_delta_skittles+abs_delta_snickers)<100)?"Nothing":(max_score == abs_delta_snickers)? 
    "Snickers": ((max_score == abs_delta_twix)? "Twix":(max_score == abs_delta_skittles)? "Skittles": "Nothing??");
    
//    Serial.print("or maybe  ");
//    Serial.println(candy);
//    
//    max_score = (abs_delta_snickers > abs_delta_skittles)?abs_delta_snickers:abs_delta_skittles;
//    max_score = (max_score > abs_delta_twix)?max_score:delta_twix;
//    candy = ((abs_delta_snickers+abs_delta_skittles+abs_delta_snickers)<100)?"Nothing":(max_score == abs_delta_snickers)? 
//    "Snickers": ((max_score == abs_delta_twix)? "Twix":(max_score == abs_delta_skittles)? "Skittles": "Nothing??");
//    
//    Serial.print("or maybe  ");
//    Serial.println(candy);
//
//  
    int lactose_detected = 0, nuts_detected = 0;
    getCandyFlags(candy, lactose_detected, nuts_detected);
  
    if (lactose_detected)
    {
      myservo1.write(90);
      myservo2.write(-30);

      Serial.println("Milk!!");
    }
    else if (nuts_detected)
    {
      myservo1.write(90);
      myservo2.write(-30);

      Serial.println("Nuts!!");
    }
    else
    {
      myservo1.write(-30);
      myservo2.write(90);

      Serial.println("safe");
    }
    
    
    }
    else
    {
    Serial.println("---Calibrating-------" + String(count) +"----");
    }
  
  
   
}


void setup() {
  Serial.begin(115200);
  //setup_display();

  tflite::InitializeTarget();
  memset(tensor_arena, 0, kTensorArenaSize*sizeof(uint8_t));
  
  // Set up logging. 
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure..
  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model provided is schema version " 
                  + String(model->version()) + " not equal "
                  + "to supported version "
                  + String(TFLITE_SCHEMA_VERSION));
    return;
  } else {
    Serial.println("Model version: " + String(model->version()));
  }
  // This pulls in all the operation implementations we need.
  static tflite::AllOpsResolver resolver;
  
  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  
  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  } else {
    Serial.println("AllocateTensor() Success");
  }

  size_t used_size = interpreter->arena_used_bytes();
  Serial.println("Area used bytes: " + String(used_size));
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model input:");
  Serial.println("dims->size: " + String(input->dims->size));
  for (int n = 0; n < input->dims->size; ++n) {
    Serial.println("dims->data[" + String(n) + "]: " + String(input->dims->data[n]));
  }

  Serial.println("Model output:");
  Serial.println("dims->size: " + String(output->dims->size));
  for (int n = 0; n < output->dims->size; ++n) {
    Serial.println("dims->data[" + String(n) + "]: " + String(output->dims->data[n]));
  }

  Serial.println("Completed tensorflow setup");
  digitalWrite(LED0, HIGH); 
  
  CamErr err = theCamera.begin(1, CAM_VIDEO_FPS_15, width, height, pixfmt);
  if (err != CAM_ERR_SUCCESS) {
    Serial.println("camera begin err: " + String(err));
    return;
  }
  err = theCamera.startStreaming(true, CamCB);
  if (err != CAM_ERR_SUCCESS) {
    Serial.println("start streaming err: " + String(err));
    return;
  }

  myservo1.attach(3);  // attaches the servo on pin 9 to the servo object
  myservo2.attach(5);  // attaches the servo on pin 9 to the servo object


}

void loop() {
}
