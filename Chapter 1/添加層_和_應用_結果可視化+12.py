
# coding: utf-8

# # add layer

# In[128]:

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ### 加入一個神經層

# In[129]:

def add_layer(inputs, in_size, out_size, activation_function=None):
    '''
    add_layer(inputs, in_size, out_size, activation_function=None)
    '''
    Weights = tf.Variable(tf.random_normal([in_size, out_size])) #[行,列]
    Biases = tf.Variable(tf.zeros([1, out_size]) + 0.1) # ML推薦Biase不為0
    Wx_plus_b = tf.matmul(inputs, Weights) + Biases # weights * data + biases
    if activation_function is None: #線性function
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# In[130]:

# -1~1的區間有300個單位(有三百個例子)
# 加上維度
x_data = np.linspace(-1,1,300)[:, np.newaxis]


# In[131]:

# 故意製造一些不規則的點在曲線周圍
# normal(loc=0.0, scale=1.0, size=None)
noise = np.random.normal(0, 0.05, x_data.shape)


# In[ ]:




# In[132]:

y_data = np.square(x_data) - 0.5 + noise


# In[133]:

### 1 input layer (input nodes by data number; 1 node)
### 1 hidden layer (10 nodes)
### 1 output (1 node)


# In[134]:

print add_layer.__doc__


# In[135]:

#tf.placeholder(tf.float32, shape=(1024, 1024))
xs = tf.placeholder(tf.float32)


# In[136]:

ys = tf.placeholder(tf.float32)


# In[137]:

# 第一層, 1個input 有10個神經元
# relu 是一個x <0, y = 0; x = 1, y=1 的線性
#l1 = add_layer(x_data, 1, 10, activation_function=tf.nn.relu)
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)


# In[138]:

# hidden layer 
prediction = add_layer(l1, 10, 1, activation_function=None)


# In[139]:

# mean(sum(spare)) ; 平均值(每個例子求合(誤差平方)
#loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data - prediton),
#                     reduction_indices=[1]))
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))


# In[140]:

#學習效率0.1, 最小化誤差
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) 


# In[141]:

init = tf.global_variables_initializer()


# In[142]:

#### 增加圖片


# In[143]:

fig = plt.figure() #圖片框


# In[144]:

ax = fig.add_subplot(1,1,1) #建立連續性的plot ; 1,1,1是編號


# In[145]:

#ax.scatter(x_data, y_data)


# In[146]:

#plt.show()


# In[147]:

#### 增加圖片 end


# In[156]:


with tf.Session() as sess:
    sess.run(init)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1) # 1 plot, 1 column, 1 row
    ax.scatter(x_data, y_data)
    plt.ion()
    plt.show()
    
    for i in range(1001):
        #training
        sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
        if i % 50 == 0:
            # to visualize the result and improvement
            try:
                ax.lines.remove(lines[0])
                #pass
            except Exception:
                pass
            #print(i, sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
            prediction_value = sess.run(prediction, feed_dict={xs:x_data})


#ani = animation.FuncAnimation(fig, animate, frames=600,interval=10, blit=True, init_func=init)
#plt.ion()
#plt.show()


# In[ ]:

## 隨著學習, 誤差有在減少


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



