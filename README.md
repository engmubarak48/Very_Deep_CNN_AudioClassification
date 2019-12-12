# Very_Deep_CNN_AudioClassification

For this task  we implemented the achitectures described in https://arxiv.org/pdf/1610.00087.pdf paper.  The achitectures described in the paper are<font color='green'> M3, M5, M11,  M18 and M34-res </font>. We followed all the recommended points in  [pitfalls](https://urbansounddataset.weebly.com/urbansound8k.html#10foldCV), however due to computational expensiveness we were not able to train as many epochs as was used in the paper. If you have the capacy and the memory you can train for more epochs to get better results.

NOTE: in the M34-res we use upsampling with mode = 'nearest' to match the size of the original input and output from the convolutional layers. For all the 5 achitectures instead using global average pool,  we used a single fully connected layer. 
Your can find the [data](https://urbansounddataset.weebly.com/)
