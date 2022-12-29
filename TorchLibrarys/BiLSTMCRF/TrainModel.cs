using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using TorchBaseLib;
using TorchLibrarys;
using TorchLibrarys.BiLSTMCRF.Config;
using TorchLibrarys.BiLSTMCRF.Data;
using TorchLibrarys.BiLSTMCRF.Model;
using TorchLibrarys.BiLSTMCRF.Utils;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.optim;
using static TorchSharp.torch.optim.lr_scheduler;

namespace ConsoleAppTorchCutWord
{
    internal class TrainModel
    {
        private Wordseg _config;
        public TrainModel(Wordseg config)
        {
            _config = config;
        }


        public void train(BilstmDataloader train_loader, BilstmDataloader dev_loader, Vocabulary vocab, BiLSTM_CRF model, Optimizer optimizer, LRScheduler scheduler, Device device,int kf_index= 0)
        {
            /*
             函数功能：
                 反复训练模型->验证参数可靠性->调整参数->继续训练模型
             细节：
                 epoch_train()：训练模型参数
                 dev()：验证模型参数可靠性，计算相关指标
                 improve_f1 ：当前模型参数得到的效果与历史最佳参数得到的效果之间的差值，如果improve_f1大于1e-5，代表当前参数比历史最佳参数更好，就应该将当前参数设置为历史最佳参数
             **/
            float best_val_f1 = 0.0f;   //历史最佳模型参数所对应的效率值，越高说明模型越好
            int patience_counter = 0;   //无法得到更优参数的连续训练模型次数
            // start training
            foreach (var epoch in Enumerable.Range(1, _config.epoch_num))//不停的进行模型训练
            {
                epoch_train(train_loader, model, optimizer, scheduler, device, epoch, kf_index);//训练模型参数
                //模型参数验证
                using (torch.no_grad())
                {
                    // dev loss calculation
                   var metric = dev(dev_loader, vocab, model, device);  //模型的验证，验证机指标计算（与真实值的偏差）
                   var val_f1 = metric["f1"];       //当前模型参数的效率值
                   var dev_loss = metric["loss"];   // 当前模型参数的损失值，后续没用到
                    if (kf_index == 0)
                        Console.WriteLine("epoch: {0}, f1 score: {1}, dev loss: {2}", epoch, val_f1, dev_loss);
                    else
                        Console.WriteLine("Kf round: {0}, epoch: {1}, f1 score: {2}, dev loss: {3}", kf_index, epoch, val_f1, dev_loss);

                    var improve_f1 = val_f1 - best_val_f1;   //best_val_f1：历史最佳参数所得到的效果值 val_f1 当前参数所得到的效果值
                    if (improve_f1 > 1e-5)       //当前参数更好，则将历史最佳参数设置为当前参数，并保存模型的参数
                    {
                        best_val_f1 = val_f1;
                        if (kf_index == 0)
                            model.save(_config.model_dir);
                        //torch.save(model, config.model_dir);
                        else
                            model.save(_config.exp_dir + $"model_{kf_index}.pth");
                            //torch.save(model, config.exp_dir + $"model_{kf_index}.pth");//模型参数保存

                        Console.WriteLine("--------Save best model!--------");

                        if (improve_f1 < _config.patience)   // 设置一个阈值5，如果连续训练5次都得不到更好的参数，则认为当前模型参数已达最优质，可以结束训练了
                        {
                            patience_counter += 1;
                        }
                        else
                        {
                            patience_counter = 0;
                        }

                    }
                    else 
                    {
                        patience_counter += 1;
                    }
                    //Early stopping and logging best f1
                    //当连续5次训练都没有得到更好的参数 或者 模型训练次数达到上届，停止训练，退出模型
                    if ((patience_counter >= _config.patience_num && epoch > _config.min_epoch_num) || epoch == _config.epoch_num)
                    { 
                        Console.WriteLine($"Best val f1: {best_val_f1}");   //满足条件，停止训练
                        break;
                    }
                }
            }

            Console.WriteLine("Training Finished!");
        }


        private void epoch_train(BilstmDataloader train_loader, BiLSTM_CRF model, Optimizer optimizer, LRScheduler scheduler, Device device,int epoch, int kf_index= 0) 
        {
            /*
            函数功能：
                1. 使用BiLSTM模型计算每个字对应四个标签的概率，比如 "我":{"B":0.2 , "E":0.3 , "M":0 , "s":0.5 } model.forward_with_crf
                2. 计算梯度   loss.backward()
                3. 根据梯度更新优化器梯度   optimizer.step();
    
            细节：
                1. 为什么要清空模型、优化器的梯度：因为上一次训练得到的梯度对本次训练没有用处，所以需要清空梯度。
            */
            model.train(); // 还没看懂
            var train_loss = 0.0;
            //得到一共需要执行多少次
            //double totalloop = Math.Ceiling((float)train_loader.Count() / _config.batch_size);
            // # tqdm 将参数装饰为迭代器；enumerate 将一个可遍历的对象组会为一个索引序列
            ProcessInfo progressBar = new ProcessInfo(Console.CursorLeft, Console.CursorTop, 50, ProgressBarType.Character);
            int i = 0;
            using (var d = torch.NewDisposeScope())
            {
                foreach (var batch_samples in train_loader)
                {
                    var (x, y, mask, lens) = batch_samples;
                    x = x.to(device);
                    y = y.to(device);
                    mask = mask.to(device);

                    model.zero_grad();    // 把上一个batch的梯度归零，上一个梯度不能影响一个计算
                   var (tag_scores, loss)= model.forward_with_crf(x, mask, y);
                    train_loss += loss.item<float>();
                    // 梯度反传
                    loss.backward();    //计算梯度
                                        // 优化更新
                    optimizer.step();   // 根据梯度，更新优化器参数
                    optimizer.zero_grad();   // 清空梯度

                    progressBar.Dispaly(Convert.ToInt32((++i / (float)train_loader.Count) * 100));
                }
                // scheduler
                scheduler.step();
                train_loss = (float)train_loss / train_loader.Count;      //计算平均损失值
                Console.WriteLine();
                if (kf_index == 0)
                {
                    Console.WriteLine("epoch: {0}, train loss: {1}", epoch, train_loss);
                }
                else
                {
                    Console.WriteLine("Kf round: {0}, epoch: {1}, train loss: {2}", kf_index, epoch, train_loss);
                }
            }
                
        }



        private Dictionary<string, float> dev(BilstmDataloader data_loader, Vocabulary vocab, BiLSTM_CRF model, Device device,string mode= "dev") 
        {
            /*
            函数功能：
                通过验证集来验证模型参数的有效性
    
            细节：
                1. 验证的思路是基于验证集的文字集合，使用模型训练出来的参数来预测其每个字所对应的标签，
                然后将其与实际标签进行比较，统计预测的正确率作为模型参数的有效性。其中true_tags是
                验证集每个字所对应的真实标签的列表，pre_tags是根据模型参数预测出来的标签的列表
                2. module.forward 函数使用LSTM模型来预测每个字对应各个标签的概率
                3. model.crf.decode 函数使用 CRF 模型来最终确定每个字所对应的标签
                4. f1_score 函数用于计算模型参数的有效性，具体来说计算了准确率、召回率和这两个比率的几何平均
                5. 模型参数的评价指标都存放在了变量 metrics 中，并作为函数输出
            */
            model.eval();
            List<List<char>> true_tags = new List<List<char>>();
            List<List<char>> pred_tags = new List<List<char>>();
            List<List<char>> sent_data = new List<List<char>>();
            float dev_losses = 0;
            using (var d = torch.NewDisposeScope())
            {
                foreach (var batch_samples in data_loader)// 读取验证集数据
                {
                    var (words, labels, masks, lens) = batch_samples;

                    words = words.to(device);
                    labels = labels.to(device);
                    masks = masks.to(device);
                    using (var y_pred = model.forward(words, training: false)) //预测验证集
                    {
                        var labels_pred = model.crf.Decode(y_pred, mask: masks);
                        //这里的extend,等同于addrange；
                        //sent_data.extend
                        //    (
                        //        [
                        //            [vocab.id2word.get(idx.item()) for i, idx in enumerate(indices) if mask[i] > 0]
                        //            for (mask, indices) in zip(masks, words)
                        //        ]
                        //   );
                        for (int i = 0; i < masks.shape[0]; i++)
                        {
                            var mask = masks[i];
                            var indices = words[i];
                            List<char> sent_datatemp = new List<char>();
                            for (int j = 0; j < indices.shape[0]; j++)
                            {
                                int idex = (int)indices[j].item<long>();
                                if (mask[j].item<byte>() > 0)
                                {
                                    sent_datatemp.Add(vocab.id2word[idex]);
                                }
                            }
                            sent_data.Add(sent_datatemp);
                        }

                        var labelstemp = labels.cpu();
                        var lenstemp = lens.data<int>().ToList();
                        int labelsrows = Convert.ToInt32(labelstemp.shape[0]);

                        //var  targets = [itag[:ilen] for itag, ilen in zip(labels.cpu().numpy(), lens)];
                        List<long[]> targets = new List<long[]>();
                        for (int i = 0; i < labelsrows; i++)
                        {
                            var itag = labelstemp[i].data<long>().ToArray();
                            var itagtemp = itag[..lenstemp[i]];
                            targets.Add(itagtemp);
                        }

                        //true_tags.extend([[vocab.id2label.get(idx) for idx in indices] for indices in targets]) ;     // 真实标签
                        foreach (var indices in targets)
                        {
                            List<char> true_tagstemp = new List<char>();
                            foreach (var idx in indices)
                            {
                                true_tagstemp.Add(vocab.id2label[(int)idx]);
                            }
                            true_tags.Add(true_tagstemp);
                        }

                        //pred_tags.extend([[vocab.id2label.get(idx) for idx in indices] for indices in labels_pred]) ;  // 预测标签
                        foreach (var indices in labels_pred)
                        {
                            List<char> pred_tagstemp = new List<char>();
                            foreach (var idx in indices)
                            {
                                pred_tagstemp.Add(vocab.id2label[idx]);
                            }
                            pred_tags.Add(pred_tagstemp);
                        }


                        // 计算梯度
                        var (_, dev_loss) =model.forward_with_crf(words, masks, labels); // BiLSMT模型，得到每个字对应的每个标签的概率
                        dev_losses += dev_loss.item<float>();
                    }
                    
                }
                if (pred_tags.Count != true_tags.Count)
                {
                    throw new Exception();
                }
                if (sent_data.Count != true_tags.Count)
                {
                    throw new Exception();
                }
                // logging loss, f1 and report
                Dictionary<string, float> metrics = new Dictionary<string, float>();       // 用于存储模型参数的评价指标
                var (f1, p, r) = Metric.f1_score(true_tags, pred_tags);   // 用于计算模型参数的评价指标
                metrics["f1"] = f1;     // p和r的几何平均
                metrics["p"] = p;       // 准确率
                metrics["r"] = r; // 召回率
                metrics["loss"] = (float)dev_losses / data_loader.Count;
                if (mode != "dev")
                {
                    Metric.bad_case(_config.case_dir, sent_data, pred_tags, true_tags);
                    Metric.output_write(_config.output_dir,sent_data, pred_tags);
                }
                return metrics;
            }
               
        }
        /// <summary>
        /// 进行数据预测
        /// </summary>
        /// <param name="data_loader"></param>
        /// <param name="vocab"></param>
        /// <param name="model"></param>
        /// <param name="device"></param>
        /// <param name="mode"></param>
        /// <returns></returns>
        private List<List<char>> forecast(BilstmDataloader data_loader, Vocabulary vocab, BiLSTM_CRF model, Device device)
        {
            model.eval();
            List<List<char>> pred_tags = new List<List<char>>();
            List<List<char>> sent_data = new List<List<char>>();
            float dev_losses = 0;
            using (var d = torch.NewDisposeScope())
            {
                foreach (var batch_samples in data_loader)// 读取验证集数据
                {
                    var (words, labels, masks, lens) = batch_samples;

                    words = words.to(device);
                    labels = labels.to(device);
                    masks = masks.to(device);
                    using (var y_pred = model.forward(words, training: false)) //预测验证集
                    {
                        var labels_pred = model.crf.Decode(y_pred, mask: masks);
                        //这里的extend,等同于addrange；
                        //sent_data.extend
                        //    (
                        //        [
                        //            [vocab.id2word.get(idx.item()) for i, idx in enumerate(indices) if mask[i] > 0]
                        //            for (mask, indices) in zip(masks, words)
                        //        ]
                        //   );
                        for (int i = 0; i < masks.shape[0]; i++)
                        {
                            var mask = masks[i];
                            var indices = words[i];
                            List<char> sent_datatemp = new List<char>();
                            for (int j = 0; j < indices.shape[0]; j++)
                            {
                                int idex = (int)indices[j].item<long>();
                                if (mask[j].item<byte>() > 0)
                                {
                                    sent_datatemp.Add(vocab.id2word[idex]);
                                }
                            }
                            sent_data.Add(sent_datatemp);
                        }

                        //pred_tags.extend([[vocab.id2label.get(idx) for idx in indices] for indices in labels_pred]) ;  // 预测标签
                        foreach (var indices in labels_pred)
                        {
                            List<char> pred_tagstemp = new List<char>();
                            foreach (var idx in indices)
                            {
                                pred_tagstemp.Add(vocab.id2label[idx]);
                            }
                            pred_tags.Add(pred_tagstemp);
                        }

                        // 计算梯度
                        var (_, dev_loss) = model.forward_with_crf(words, masks, labels); // BiLSMT模型，得到每个字对应的每个标签的概率
                        dev_losses += dev_loss.item<float>();
                    }

                }
                return pred_tags;
            }

        }
        public (float,float) test(string dataset_dir, Vocabulary vocab, Device device,int kf_index= 0)
        {
            //test model performance on the final test set
            var dicdata = JsonConvert.DeserializeObject<Dictionary<string, List<List<char>>>>(File.ReadAllText(dataset_dir));
            //data = np.load(dataset_dir, allow_pickle = True);
            var word_test = dicdata["words"];
            var label_test = dicdata["labels"];
            // build dataset
            var test_dataset =new SegDataset(word_test, label_test, vocab, _config.label2id);
            // build data_loader;
            var test_loader = new BilstmDataloader(test_dataset, batchSize: _config.batch_size,collate_fn: test_dataset.collate_fn, shuffler:false);
            BiLSTM_CRF model = null;
            if (kf_index == 0)
            {
                model = BiLSTM_CRF.Load(_config.model_dir) as BiLSTM_CRF;
                model.to(device);
                Console.WriteLine("--------Load model from {0}--------", _config.model_dir);
            }
            else 
            {
                model = BiLSTM_CRF.Load(_config.exp_dir + $"model_{kf_index}.pth") as BiLSTM_CRF;
                model.to(device);
                Console.WriteLine("--------Load model from {0}--------", _config.exp_dir + $"model_{kf_index}.pth");
            }
            var metric = dev(test_loader, vocab, model, device, mode:"test");
            var f1 = metric["f1"];
            var p = metric["p"];
            var r = metric["r"];
            var test_loss = metric["loss"];
            if (kf_index == 0)
            {
                Console.WriteLine($"final test loss: {0}, f1 score: {1}, precision:{2}, recall: {3}",test_loss, f1, p, r);
            }
            else 
            {
                Console.WriteLine($"Kf round: {0}, final test loss: {1}, f1 score: {2}, precision:{3}, recall: {4}", kf_index, test_loss, f1, p, r);
            }
        
            return (test_loss, f1);

        }


        public (List<List<char>>, List<List<char>>) test_content(string content, Vocabulary vocab, Device device, int kf_index = 0)
        {
            var contentlist = Regex.Split(content, @"(\r\n|\s|；|，|。|？)");

            List<List<char>> contentchars = new List<List<char>>();
            List<List<char>> labelchars = new List<List<char>>();
            foreach (string linetxt in contentlist)
            {
                string linestr= linetxt.Trim();
                if (string.IsNullOrWhiteSpace(linestr))
                    continue;
                List<char> labeltemp = new List<char>();
                for (int i = 0; i < linestr.ToList().Count; i++)
                {
                    labeltemp.Add('S');
                }
                labelchars.Add(labeltemp);
                contentchars.Add(linestr.ToList());
            }
            var word_test = contentchars;
            // build dataset
            var test_dataset = new SegDataset(word_test, labelchars, vocab, _config.label2id);
            // build data_loader;
            var test_loader = new BilstmDataloader(test_dataset, batchSize: _config.batch_size, collate_fn: test_dataset.collate_fn, shuffler: false);
            BiLSTM_CRF model = new BiLSTM_CRF(embedding_size: _config.embedding_size,    //初始化模型
                       hidden_size: _config.hidden_size,
                       vocab_size: vocab.vocab_size(),
                       target_size: vocab.label_size(),
                       num_layers: _config.lstm_layers,
                       lstm_drop_out: _config.lstm_drop_out,
                       nn_drop_out: _config.nn_drop_out,
                       device: device);
            if (kf_index == 0)
            {
                model.load(_config.model_dir);
                Console.WriteLine("--------Load model from {0}--------", _config.model_dir);
            }
            else
            {
                model.load(_config.exp_dir + $"model_{kf_index}.pth");
                Console.WriteLine("--------Load model from {0}--------", _config.exp_dir + $"model_{kf_index}.pth");
            }
            var pre_tags = forecast(test_loader, vocab, model, device);
             
            return (contentchars, pre_tags);
        }


        ///// <summary>
        ///// 加载模型
        ///// </summary>
        ///// <param name="model_dir"></param>
        ///// <param name="device"></param>
        ///// <returns></returns>
        //private T load_model<T>(string model_dir,Device device) where T: nn.Module
        //{
        //    // Prepare model
        //    var model = T.Load(model_dir);
        //    //var model = torch.load(model_dir);
        //    model.to(device);
        //    Console.WriteLine("--------Load model from {0}--------",model_dir);
        //    return model;
        //}
    
    }
}
