using TorchSharp;
using static TorchSharp.torch.utils.data;
using static TorchSharp.torch;
using TorchSharp.Modules;
using System.Collections.Generic;
using System.Linq;
using System;
using TorchLibrarys.BiLSTMCRF.Utils;

namespace TorchLibrarys.BiLSTMCRF.Data
{
    //public class SegDataset:Dataset<(Tensor, Tensor, Tensor, Tensor)>
    public class SegDataset:Dataset<List<List<int>>>
    {
        public Vocabulary vocab { get; private set; }
        public List<(int[], int[])> dataset { get; private set; }
        public Dictionary<char, int> _label2id { get; private set; }

        public override long Count => dataset.Count;

        public SegDataset(List<List<char>> words, List<List<char>> labels, Vocabulary vocab, Dictionary<char, int> label2id)
        {
            this.vocab = vocab;
            this.dataset = this.preprocess(words, labels);
            this._label2id = label2id;
        }
        private List<(int[], int[])> preprocess(List<List<char>> words, List<List<char>> labels) 
        {
            //convert the data to ids
            var processed = new List<(int[], int[])>();
            foreach (var (word, label) in words.Zip(labels))
            {
                var word_id = word.Select(u_ => this.vocab.word_id(u_)).ToArray();
                var label_id = label.Select(l_=> this.vocab.label_id(l_)).ToArray();
                //var label_id = [this.vocab.label_id(l_) for l_ in label];
                processed.Add((word_id, label_id));
            }
            Console.WriteLine("-------- Process Done! --------");
            return processed;
        }

        //public override (Tensor, Tensor, Tensor, Tensor) GetTensor(long index)
        public override List<List<int>> GetTensor(long index)
        {
            var row = dataset[(int)index];
            return new List<List<int>>() { row.Item1.ToList(), row.Item2.ToList() };
        }

        public (Tensor word_ids, Tensor label_ids, Tensor input_mask) get_long_tensor(List<List<int>> words, List<List<int>> labels,int batch_size)
        {
            var token_len = labels.Select(x => x.Count).Max();
            //var token_len = max([len(x) for x in labels])

            //有问题
            //Tensor word_tokens = torch.as_tensor(new long[] { batch_size, token_len }, torch.ScalarType.Float64).fill_(0);
            //Tensor label_tokens = torch.as_tensor(new long[] { batch_size, token_len }, torch.ScalarType.Float64).fill_(0);
            //Tensor mask_tokens = torch.as_tensor(new long[] { batch_size, token_len }, torch.ScalarType.Byte).fill_(0);

            int totalsize = batch_size * token_len;
            Tensor word_tokens = torch.zeros(new long[] { batch_size, token_len }, torch.ScalarType.Int64);
            Tensor label_tokens = torch.zeros(new long[] { batch_size, token_len }, torch.ScalarType.Int64);
            Tensor mask_tokens = torch.zeros(new long[] { batch_size, token_len }, torch.ScalarType.Byte);

            //Tensor word_tokens = torch.LongTensor(batch_size, token_len).fill_(0);
            //Tensor label_tokens = torch.LongTensor(batch_size, token_len).fill_(0);
            //Tensor mask_tokens = torch.ByteTensor(batch_size, token_len).fill_(0);
            var mergeCollecon= words.Zip(labels);
            int i = 0;
            foreach (var (j,s) in words.Zip(labels))
            {
                //py写法
                //word_tokens[i, :len(s[0])] = torch.LongTensor(s[0])
                //label_tokens[i, :len(s[1])] = torch.LongTensor(s[1])
                //mask_tokens[i, :len(s[0])] = torch.tensor([1] * len(s[0]), dtype = torch.uint8)

                //charp写法
                word_tokens[i,..(j.Count)] = torch.LongTensor(j.ToArray());
                label_tokens[i,..(s.Count)] = torch.LongTensor(s.ToArray());
                mask_tokens[i,..(j.Count)] = torch.ones(j.Count, ScalarType.Byte);

                //word_tokens.index_put_(torch.LongTensor(j.ToArray()), torch.TensorIndex.Single(i));
                //label_tokens.index_put_(torch.LongTensor(s.ToArray()), torch.TensorIndex.Single(i));
                //mask_tokens.index_put_(torch.ones(j.Count, ScalarType.Int8), torch.TensorIndex.Single(i));
                i++;
            }
            //var new_word_tokens = word_tokens.reshape(batch_size, token_len);
            //var new_label_tokens = label_tokens.reshape(batch_size, token_len);
            //var new_mask_tokens = mask_tokens.reshape(batch_size, token_len);

            return (word_tokens, label_tokens, mask_tokens);

        }

        //public (Tensor, Tensor, Tensor, Tensor) collate_fn(IEnumerable<List<List<int>>> batch, Device device) 
        public (Tensor, Tensor, Tensor, Tensor) collate_fn(IEnumerable<List<List<int>>> batch, Device device) 
        {
            var words= batch.Select(x => x[0]).ToList();
            var labels= batch.Select(x => x[1]).ToList();
            var lens = labels.Select(x => x.Count).ToList();
            int  batch_size = batch.Count();
            var (word_ids, label_ids, input_mask) = this.get_long_tensor(words, labels, batch_size);
            return (word_ids, label_ids, input_mask,torch.as_tensor(lens));
        }
    }


    public class SegRowData 
    {
    
    }
}
