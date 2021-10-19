# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 14:05:53 2021

@author: innat
"""
import tensorflow as tf 

class BrainTumorModel3D(tf.keras.Model):
    def __init__(self, model, n_gradients=1):
        super(BrainTumorModel3D, self).__init__()
        self.model = model
        self.n_gradients = tf.constant(n_gradients, dtype=tf.int32)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), 
                                                  trainable=False) for v in self.model.trainable_variables]
    
    def train_step(self, data):
        self.n_acum_step.assign_add(1)
        
        images, labels = data
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            loss = self.compiled_loss(labels,
                                      predictions,
                                      regularization_losses=[self.reg_l2_loss()])
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Accumulate batch gradients
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])
 
        # If n_acum_step reach the n_gradients then we apply accumulated gradients 
        # to update the variables otherwise do nothing
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: None)
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, labels = data
        predictions = self.model(images, training=False)
        loss = self.compiled_loss(labels, predictions, 
                                  regularization_losses=[self.reg_l2_loss()])
        self.compiled_metrics.update_state(labels, predictions)
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, *args, **kwargs):
        return self.model(inputs)
    
    def reg_l2_loss(self, weight_decay = 1e-5):
        return weight_decay * tf.add_n([
            tf.nn.l2_loss(v)
            for v in self.model.trainable_variables
        ])
    
    def apply_accu_gradients(self):
        # apply accumulated gradients
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, 
                                           self.model.trainable_variables))

        # reset
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(
                tf.zeros_like(
                    self.model.trainable_variables[i], dtype=tf.float32)
            )
