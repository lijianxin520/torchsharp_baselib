using System;
using System.IO;
using TorchBaseLib;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchLibrarys.BiLSTMCRF.Model
{
    /// <summary>
    /// https://github.com/hemingkx/WordSeg/blob/main/BiLSTM-CRF/config.py
    /// https://github.com/WhiteGive-Boy/CWS-Hmm_BiLSTM-CRF/blob/master/LSTM/train.py
    /// https://github.com/hemingkx/WordSeg/blob/main/BiLSTM-CRF/model.py
    /// https://github.com/jasoncao11/nlp-notebook
    /// https://github.com/Riccorl/chinese-word-segmentation-pytorch/blob/6bb4fd4783dbb205181172f1bee774687d69c903/cws/dataset.py#L2
    /// https://github.com/hemingkx/WordSeg/blob/main/BiLSTM-CRF/data_loader.py
    /// https://github.com/VenRay/ChineseSegmentationPytorch/blob/master/build.py
    /// </summary>
    public class BiLSTM_CRF : Module<Tensor,Tensor>
    {

        //string START_TAG = "<START>";
        //string STOP_TAG = "<STOP>";
        //int EMBEDDING_DIM = 5;
        //int HIDDEN_DIM = 4;


        private Embedding word_embeds;
        private LSTM bilstm;
        private Linear classifier;
        private Dropout dropout;
        private long target_size;
        private long nn_drop_out;
        /// <summary>
        /// CRF方法
        /// </summary>
        public TorchSharpCrf crf { get; }

        /// <summary>
        /// 实例化BiLstm
        /// </summary>
        /// <param name="embedding_size"></param>
        /// <param name="hidden_size"></param>
        /// <param name="vocab_size"></param>
        /// <param name="target_size"></param>
        /// <param name="num_layers"></param>
        /// <param name="lstm_drop_out"></param>
        /// <param name="nn_drop_out"></param>
        /// <param name="device"></param>
        /// <param name="name"></param>
        public BiLSTM_CRF(long embedding_size, long hidden_size, long vocab_size, long target_size, int num_layers, double lstm_drop_out,long nn_drop_out, Device device) : base("bilstm")
        {
            this.nn_drop_out = nn_drop_out;
            this.word_embeds = nn.Embedding(vocab_size, embedding_size);
            this.bilstm = nn.LSTM(
                    inputSize:embedding_size,
                    hiddenSize:hidden_size,
                    batchFirst:true,
                    numLayers:num_layers,
                    dropout: num_layers>1?lstm_drop_out:0,
                    bidirectional:true
                    );
            if (nn_drop_out > 0)
            {
                this.dropout = nn.Dropout(nn_drop_out);
            }
            // 将模型加载到CPU中
            //加载到指定驱动器
            
            this.target_size = target_size;
            this.classifier = nn.Linear(hidden_size * 2, target_size);
            this.crf = new TorchSharpCrf(target_size, batch_first: true);
            this.crf.to(device);
            //创建模型一定要进行注册组件；否则计算会有问题；
            RegisterComponents();
            this.to(device);
        }
        /// <summary>
        /// 前向传播，梯度计算
        /// </summary>
        /// <param name="unigrams"></param>
        /// <param name="training"></param>
        /// <returns></returns>
        public  Tensor forward(Tensor unigrams, bool training=true)
        {
            //this.word_embeds
           var uni_embeddings = this.word_embeds.forward(unigrams);   // 将字编码，从而节约存储空间，如 "你"编码为[0.2,0.1]
           var (sequence_output, _,_) = this.bilstm.forward(uni_embeddings);        // 使用LSTM模型得到每个字对应四种标签的概率
            if (training && this.nn_drop_out > 0)
            {
                sequence_output = this.dropout.forward(sequence_output);
            }
            var tag_scores = this.classifier.forward(sequence_output);   // 转换数据维度，因为BiLSTM模型可以是n-m模型，即输入参数维度为n，输出参数维度为m，故需要转换数据维度
            return tag_scores;
        }

        public override Tensor forward(Tensor input1)
        {
            throw new NotImplementedException();
        }

        public (Tensor, Tensor) forward_with_crf(Tensor unigrams, Tensor input_mask, Tensor input_tags)
        {
            /**
             * 函数功能：
                1. 使用BiLSTM模型计算每个字对应的4个标签的概率 self.forware
                2. 使用crf算法计算损失值 self.crf
             */

            var tag_scores = this.forward(unigrams);     // BiLSMT模型，得到每个字对应的每个标签的概率
            var loss = this.crf.forward(tag_scores, input_tags, input_mask) * (-1);
            return (tag_scores, loss);
        }
    }
}
