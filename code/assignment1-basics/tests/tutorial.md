
# 需要依次通过的作业文件顺序说明

本文件夹下的测试文件需要按如下顺序依次通过：


3. test_model.py ✅      —— 然后实现模型相关内容
4. test_nn_utils.py      —— 最后完成神经网络工具相关内容

请按照上述顺序逐步完成和通过每个测试文件。


1. Byte-pair encoding (BPE) tokenizer (§2)
    1.1 test_train_bpe.py ✅  —— 先完成BPE训练相关内容
    1.2 test_tokenizer.py ✅  —— 再完成分词器相关内容
2. Transformer language model (LM) (§3)
    2.1 
    2.2 
3. The cross-entropy loss function and the AdamW optimizer (§4)
4. The training loop, with support for serializing and loading model and optimizer state (§5)