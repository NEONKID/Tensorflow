# -*- coding: utf-8 -*-
import tensorflow as tf

# 변수를 선언, 그리고 0으로 초기화,,
state = tf.Variable(0, name="counter")

# state 변수에 1을 더할 operation 생성,,
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# 그래프를 사용할 경우, 처음에는 변수를 초기화 해줘야 함,,
init_op = tf.initialize_all_variables()

# 그래프를 띄우고, Operation 실행,,
with tf.Session() as sess:
    sess.run(init_op) # 초기화 Operation,,
    print sess.run(init_op)
    
    for _ in range(3):
        sess.run(update)
        print sess.run(state)
