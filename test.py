# -*-coding:utf-8 -*-
import tensorflow as tf  # 0.12
from tensorflow.models.rnn.translate import seq2seq_model
import os
import numpy as np
import jieba

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

train_encode_vocabulary = 'train_encode_vocabulary'
train_decode_vocabulary = 'train_decode_vocabulary'

#读取字典中数据
def read_vocabulary(input_file):
    tmp_vocab = []
    with open(input_file, "r") as f:
        tmp_vocab.extend(f.readlines())
    tmp_vocab = [line.strip().decode('utf-8') for line in tmp_vocab]  #删除词典中的空白符（包括'\n','\r','\t',' ')
    vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)])
    return vocab, tmp_vocab


vocab_en, _, = read_vocabulary(train_encode_vocabulary)
print (vocab_en)
_, vocab_de, = read_vocabulary(train_decode_vocabulary)
print (vocab_de)

# 词汇表大小50000
vocabulary_encode_size = 50000
vocabulary_decode_size = 50000

buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]
layer_size = 256  # 每层大小
num_layers = 3  # 层数
batch_size = 1

model = seq2seq_model.Seq2SeqModel(source_vocab_size=vocabulary_encode_size, target_vocab_size=vocabulary_decode_size,
                                   buckets=buckets, size=layer_size, num_layers=num_layers, max_gradient_norm=5.0,
                                   batch_size=batch_size, learning_rate=0.5, learning_rate_decay_factor=0.99,
                                   forward_only=True)
model.batch_size = 1

with tf.Session() as sess:
    # 恢复前一次训练
    ckpt = tf.train.get_checkpoint_state('.')
    if ckpt != None:
        print(ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("没找到模型")

    while True:
        input_string = raw_input('me > ')
        # 退出
        if input_string == 'quit':
            exit()

        input_string_vec = []
        for words in jieba.cut(input_string.strip().decode('utf-8')):
            input_string_vec.append(vocab_en.get(words, UNK_ID))
        print (input_string_vec )
        bucket_id = min([b for b in range(len(buckets)) if buckets[b][0] > len(input_string_vec)])
        encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(input_string_vec, [])]},
                                                                         bucket_id)
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
        print (output_logits,len(output_logits),len(output_logits[0][0]))
        # 将最大值作为输出
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        print (outputs)
        if EOS_ID in outputs:
            outputs = outputs[:outputs.index(EOS_ID)]
        print (outputs)
        # 将词典中的词进行组合，得到回答
        response = "".join([tf.compat.as_str(vocab_de[output]) for output in outputs])
        print('AI > ' + response)
