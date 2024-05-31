import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
import numpy as np

training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
target_data = np.array([[0],[1],[1],[0]], "float32")
test_data_X = np.array([[0,0],[1,1],[1,0],[0,1]], "float32")
test_data_Y = np.array([[0],[1],[1],[0]], "float32")

model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

model.fit(training_data, target_data, epochs=200, verbose=2)

model.summary()

print(model.predict(training_data))

print("학번 : 20190831")
print("이름 : 신지섭")
print("[1]학습")
print("[2]weight 저장")
print("[3]weight 읽기")
print("[4]테스트")
print("[5]종료")


while(1):
   sel = int(input("입력 :"))
   match sel:
       case 1:
           model.fit(training_data, target_data, epochs=200, verbose=2)
           model.summary()
       case 2:
            model.save_weights("testweight.weights.h5")
       case 3:
           model.load_weights("testweight.weights.h5")
       case 4:
           print(model.predict(test_data_X))
       case 5: 
           exit()
       case _:
           print("다시 입력하세요")
    
