# -*- coding: utf-8 -*- 

'''
Created on Jul 27, 2016

@author: neonkid
'''

import tensorflow as tf

# training data,,
x_data = [1, 2, 3]
y_data = [1, 2, 3]

'''
Try to find values for W and b that compute y_data = W * x_data + b
(We know that W should be 1 and b 0, but Tensorflow will
figure that out for us. )

W와 b는 tensorflow가 지정하는 variable로 정의한다.
그래야만 기계가 학습해서 변동될 값을 저장할 수 있기 때문.

W가 1, b가 0이 되면 저 위 데이터가 매핑될 수 있는 가장 알맞는 
Hyperthesis가 되는데, 이를 기계가 할 수 있는지 없는지를 테스트,,

테스트를 위해, Variable 값을 random하게 준다. 그래서 기계가 자동으로
찾는지 어떤지를 확인,,

범위는 -1 ~ 1까지 준다.

'''
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis - 모델 적용,,
hypothesis = W * X + b     # H(x) = Wx + b


'''
square 함수는 제곱을 구하는 기본 함수이고, 
reduce_mean은 평균을 구하는 함수이다.

참고로, tensorflow는 sess.run을 통해 수행하지 않는 이상
아직 계산 되는 것은 아니다. 우리가 지금 하는 작업은 
모두 Operation일 뿐이라는 것을 잊지말자.

'''
# Simplified cost function - Cost function 모델,, 
cost = tf.reduce_mean(tf.square(hypothesis - Y))   # cost(W,b) = 예측 값 - 실제 값 y 의 제곱(square)

'''
Blackbox :

GradientDescentOptimizer 함수를 이용해, Minimize를 할 수 있다.
해당 함수에 대한 분석은 스스로..

'''
# Minimize - 최적화,,
a = tf.Variable(0.1)    # Learning rate, alpha, 바뀌는 속도,,
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

'''
변수의 초기화 수행,,
위에 있는 것들은 모두 Operation이기 때문에 
tensorflow에서 변수를 초기화 하기 위해, 아래 구문 사용

이 구문 안 주면 오류남,,

'''
# Before starting, initalize the variables, We will 'run' this first
init = tf.initialize_all_variables()

# Launch the graph,,
sess = tf.Session()
sess.run(init)  # 변수 초기화, 진짜 실행,,

# Fit the Line
for step in xrange(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W), sess.run(b) 
