using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchLibrarys.BiLSTMCRF.Config;
using static TorchSharp.torch;
using TorchSharp;
using TorchLibrarys.BiLSTMCRF.Data;
using TorchLibrarys.BiLSTMCRF.Utils;
using static TorchSharp.torch.optim.lr_scheduler;
using ConsoleAppTorchCutWord;
using System.IO;

namespace TorchLibrarys.BiLSTMCRF
{
    public class BiLstmRun
    {
        private Wordseg _config;
        public BiLstmRun(Wordseg config)
        {
            _config = config;
        }

        public void Simple_run()
        {
            /*
             函数功能：
                 对数据进行预处理，建立样本集和验证集，并将所有的字符找到其对应的标签，建立一一对应关系

             细节：
                 data_process(): 为样本数据集中的词组添加BMES标签
                 get_vocab(): 构建词表 self.word2id & self.id2word，具体详见函数解释
                 dev_split()：按照9:1比例切分样本集和验证集
                 run():运行模型、训练、测试
             **/
            // 设置gpu为命令行参数指定的id
            //目前只有CPU；
            Device device = null;
            //// This worked on a GeForce RTX 2080 SUPER with 8GB, for all the available network architectures.
            //// It may not fit with less memory than that, but it's worth modifying the batch size to fit in memory.
            //torch.cuda.is_available() ? torch.CUDA :
            //torch.CPU;
            if (_config.gpu != "")
                device = torch.device("cuda:{config.gpu}");
            else
                device = torch.device("cpu");    //用cpu跑模型

            Console.WriteLine("device: {0}", device);

            // 处理数据，分离文本和标签
            var processor = new DataProcessor(_config.data_dir, _config.files.ToArray());   // 找到数据集
            processor.data_process();                 //给样本集添加BMES标签



            // 建立词表
            var vocab = new Vocabulary(_config); //key=id(0-500) value="word"
            vocab.get_vocab();  // 构建词表 self.word2id & self.id2word，具体详见函数解释


            // 分离出验证集
            var (word_train, word_dev, label_train, label_dev) = dev_split(_config.train_dir, _config.dev_split_size);// 参数为训练集
            // simple run without k-fold
            Run(word_train, label_train, word_dev, label_dev, vocab, device);//运行训练集和测试集

        }

        /// <summary>
        /// 进行模型数据验证
        /// </summary>
        /// <param name="content"></param>
        public void DataEval(string content) 
        {
            Device device = torch.device("cpu");    //用cpu跑模型
            // 建立词表
            var vocab = new Vocabulary(_config); //key=id(0-500) value="word"
            vocab.get_vocab();  // 构建词表 self.word2id & self.id2word，具体详见函数解释
            //train and test
            TrainModel trainCls = new TrainModel(_config);
            var result=trainCls.test_content(content, vocab, device);
            foreach (var (item,item2) in result.Item1.Zip(result.Item2))
            {
                for (int i = 0; i < item.Count; i++)
                {
                    Console.Write($"{item[i]}[{item2[i]}]");
                }
            }
            Console.WriteLine();
        }


        /// <summary>
        /// 执行运行
        /// </summary>
        /// <param name="word_train"></param>
        /// <param name="label_train"></param>
        /// <param name="word_dev"></param>
        /// <param name="label_dev"></param>
        /// <param name="vocab"></param>
        /// <param name="device"></param>
        /// <param name="kf_index"></param>
        private (float, float) Run(List<List<char>> word_train, List<List<char>> label_train, List<List<char>> word_dev, List<List<char>> label_dev, Vocabulary vocab, Device device, int kf_index = 0)
        {
            /*
                函数功能： 
                    1. 建立样本集和测试集的迭代器  train_loader / dev_loader
                    2. 建立模型并映射到cpu设备上  module
                    3. 建立优化器 optimizer
                    4. 建立调整学习率的方法 scheduler
                    5. 归一化处理crf模型中的参数  model.crf.parameters
                    6. 训练、验证模型，并调整参数 train
                    7. 测试最终的模型   test
                **/
            //测试集   验证集  词表 设备
            //build dataset
            var train_dataset = new SegDataset(word_train, label_train, vocab, _config.label2id);// 训练集，包含了字列表和标签列表，这两个列表一一对应
            var dev_dataset = new SegDataset(word_dev, label_dev, vocab, _config.label2id);      // 验证集
            //build data_loader
            // 创造一个迭代器，方便接下来的模型迭代的访问训练集
            //var train_loader = new data.DataLoader<List<List<int>>,(Tensor, Tensor, Tensor, Tensor)>(train_dataset, batchSize: _config.batch_size,collate_fn: train_dataset.collate_fn, shuffle: true);
            var train_loader = new BilstmDataloader(train_dataset, batchSize: _config.batch_size, collate_fn: train_dataset.collate_fn, shuffler: true);
            // 创造一个迭代器，方便接下来的模型迭代的访问验证集
            var dev_loader = new BilstmDataloader(dev_dataset, batchSize: _config.batch_size, collate_fn: dev_dataset.collate_fn, shuffler: true);
            // model
            var model = new Model.BiLSTM_CRF(embedding_size: _config.embedding_size,    //初始化模型
                       hidden_size: _config.hidden_size,
                       vocab_size: vocab.vocab_size(),
                       target_size: vocab.label_size(),
                       num_layers: _config.lstm_layers,
                       lstm_drop_out: _config.lstm_drop_out,
                       nn_drop_out: _config.nn_drop_out,
                       device: device);

            //初始化模型，把模型传入cpu
            //optimizer  betas:config.betas
            var optimizer = optim.Adam(model.parameters(), lr: _config.lr);//optimizer是一个优化器，可以保存当前的参数，并根据计算得到的梯度来更新参数
            //关于优化器可以参考 https://blog.csdn.net/KGzhang/article/details/77479737
            var scheduler = StepLR(optimizer, step_size: _config.lr_step, gamma: _config.lr_gamma);
            //用于等间隔调整学习率的方法，每三个epoch调整一次学习率,调整倍数gamma是0，5；学习率衰减，参数变化幅度变小，便于收敛，
            //关于学习率调整函数可以参考 https://zhuanlan.zhihu.com/p/69411064
            //how to initialize these parameters elegantly

            foreach (var p in model.crf.parameters())
            { //归一化crf模型中的参数，将参数的值放缩到[-1,1]之间
                torch.nn.init.uniform_(p, -1, 1);
            }
            //train and test
            TrainModel trainCls = new TrainModel(_config);
            trainCls.train(train_loader, dev_loader, vocab, model, optimizer, scheduler, device, kf_index);// 模型训练、验证


            using (torch.no_grad())
            {
                //test on the final test set
                var (test_loss, f1) = trainCls.test(_config.test_dir, vocab, device, kf_index);
                return (test_loss, f1);
            }
        }


        private (List<List<char>> x_t, List<List<char>> x_d, List<List<char>> y_t, List<List<char>> y_d) dev_split(string dataset_dir, double dev_split_size)
        {
            /*
            函数功能： 将数据按照9:1 切分为训练集和测试集

            返回值
                x_train： 训练集——字集合  距离
                y_train:  训练集——标签集合
                x_dev:    测试集——字集合
                y_dev     测试集——标签集合
            **/
            string data = File.ReadAllText(dataset_dir, Encoding.UTF8);
            var dicdata = JsonConvert.DeserializeObject<Dictionary<string, List<List<char>>>>(data);
            //data = np.load(dataset_dir, allow_pickle = True);
            var words = dicdata["words"];
            var labels = dicdata["labels"];
            var (x_train, x_dev, y_train, y_dev) = train_test_split(words, labels, dev_split_size, 0);
            return (x_train, x_dev, y_train, y_dev);
        }
        /// <summary>
        /// 生成分割数据
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="words"></param>
        /// <param name="labels"></param>
        /// <param name="test_size"></param>
        /// <param name="random_state"></param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        private (T x_t, T x_d, T y_t, T y_d) train_test_split<T>(T words, T labels, double test_size, int random_state) where T : List<List<char>>, new()
        {
            T x_t = new T();
            T x_d = new T();
            T y_t = new T();
            T y_d = new T();
            //获取所有数量
            int count = words.Count();
            int count_l = labels.Count();
            if (count != count_l)
            {
                throw new Exception("内容数量与标签数量不对等！");
            }
            x_t.AddRange(words);
            y_t.AddRange(labels);
            /**
             *  // Set up the indices array.
            Random rnd = new Random();
            var indices = !shuffle ?
                Enumerable.Range(0, count).ToArray() :
                Enumerable.Range(0, count).OrderBy(c => rnd.Next()).ToArray();
             */
            int testcount = (int)(count * test_size);
            Random rnd = new Random();
            var indices = Enumerable.Range(0, testcount).OrderBy(c => rnd.Next(0, count)).ToArray();
            foreach (int idx in indices)
            {
                x_d.Add(x_t[idx]);
                y_d.Add(y_t[idx]);
            }
            foreach (int idx in indices)
            {
                x_t.RemoveAt(idx);
                y_t.RemoveAt(idx);
            }
            //var y_t= words.ToArray().GetValue(indices) as T;
            //var y_d = labels.ToArray().GetValue(indices) as T;
            return (x_t, x_d, y_t, y_d);

        }

    }
}
