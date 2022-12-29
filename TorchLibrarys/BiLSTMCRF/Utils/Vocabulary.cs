using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TorchLibrarys.BiLSTMCRF.Config;
using static TorchSharp.torch.utils;

namespace TorchLibrarys.BiLSTMCRF.Utils
{
    /// <summary>
    /// 构建词表
    /// </summary>
    public class Vocabulary: VocabularyModel
    {
        public string data_dir { get; set; }
        public List<string> files { get; set; }
        public string vocab_path { get; set; }
        public int max_vocab_size { get; set; }
        //public Dictionary<char,int> word2id { get; set; }
        //public Dictionary<int, char> id2word { get; set; }
        public Dictionary<char, int> label2id { get; set; }
        public Dictionary<int, char> id2label { get; set; }
        public Vocabulary(Wordseg config)
        {
            this.data_dir = config.data_dir;
            this.files = config.files;
            this.vocab_path = config.vocab_path;
            this.max_vocab_size = config.max_vocab_size;
            this.word2id = new Dictionary<char, int>();
            this.id2word = null;
            this.label2id = config.label2id;
            this.id2label = config.id2label;
        }
        public int vocab_size() 
        {
            return this.word2id.Count;
        }
        public int label_size() 
        {
            return this.label2id.Count;
        }
        /// <summary>
        /// 获取词的id
        /// </summary>
        /// <returns></returns>
        public int word_id(char word) 
        {
            return this.word2id[word];
        }
        /// <summary>
        /// 获取id对应的词
        /// </summary>
        /// <returns></returns>
        public char id_word(int idx) 
        {
            return this.id2word[idx];
        }
        /// <summary>
        /// 获取label的id
        /// </summary>
        /// <returns></returns>
        public int label_id(char word) 
        {
            return this.label2id[word];
        }
        /// <summary>
        /// 获取id对应的词
        /// </summary>
        /// <returns></returns>
        public char id_label(int idx) 
        {
            return this.id2label[idx];
        }
        public void get_vocab() 
        {
            /*
            函数功能：
                进一步处理，将word和label转化为id
                word2id: dict,每个字对应的序号
                idx2word: dict,每个序号对应的字
            细节：
                如果是第一次运行代码则可直接从65行开始看就行（第一次运行没有处理好的vocab可供直接读取）
                该函数统计样本集中每个字出现的次数，并制作成词表，然后按照出现次数从高到低排列，并给每个字赋予一个唯一的id（出现次数越多id越小）
            
                最终 self.word2id 形如 {"我":1 , "你":2 , "他":3 ...}

            输出：
                保存为二进制文件
            **/
            // 如果有处理好的，就直接load
            if (File.Exists(this.vocab_path))
            {
                var data=JsonConvert.DeserializeObject<VocabularyModel>(File.ReadAllText(vocab_path));
                //data = np.load(self.vocab_path, allow_pickle = True)
                //'[()]'将array转化为字典
                this.word2id =data.word2id;
                this.id2word = data.id2word;
                Console.WriteLine("-------- Vocabulary Loaded! --------");
                return;
            }
            //如果没有处理好的二进制文件，就处理原始的npz文件
            var word_freq =new Dictionary<char,int>();
            foreach (var file in this.files)
            {
               var dicdata= JsonConvert.DeserializeObject<Dictionary<string, List<List<char>>>>(File.ReadAllText(this.data_dir +"/"+ file + ".npz"));
                //data = np.load(this.data_dir + str(file) + '.npz', allow_pickle = True)   // 打开之前压缩好的 词-标签 的.npz文件
               var word_list = dicdata["words"];        // 读取其中的词列表
                // 常见的单词id最小
                foreach (var line in word_list)//按行读取
                {
                    foreach (var ch in line)//按字读取 
                    {
                        if (word_freq.ContainsKey(ch))//统计每个字出现的频率，并统计在word_freq中
                        {
                            word_freq[ch] += 1;
                        }
                        else
                        {
                            word_freq[ch] = 1;
                        }
                    }
                }
            }
            int index = 0;
            // 按照字的出现频率降序排列
            var sorted_word = word_freq.OrderByDescending(x => x.Value).ToDictionary(k => k.Key, v => v.Value);
            //构建word2id字典
            foreach (var elem in sorted_word)
            {
                this.word2id[elem.Key] = index;// 出现频率越高的字出现在越前面
                index += 1;
                if (index >= this.max_vocab_size)
                    break;
            }
            //id2word保存
            this.id2word = this.word2id.ToDictionary(k => k.Value, v => v.Key);
            //this.id2word = { _idx: _word for _word, _idx in list(this.word2id.items())};
            //保存为二进制文件
            //var savedata= new Dictionary<string, object>() { { "word2id", this.word2id }, { "id2word", this.id2word } };
            var savedata = new VocabularyModel()
            {
                word2id = this.word2id,
                id2word = this.id2word
            };
            File.WriteAllText(this.vocab_path, JsonConvert.SerializeObject(savedata),System.Text.Encoding.UTF8);
            //np.savez_compressed(this.vocab_path, word2id = this.word2id, id2word = this.id2word);
            Console.WriteLine("-------- Vocabulary Build! --------");
        }
    }



    public class VocabularyModel 
    {
        public Dictionary<char, int> word2id { get; set; }
        public Dictionary<int, char> id2word { get; set; }
        //public Dictionary<char, int> label2id { get; set; }
        //public Dictionary<int, char> id2label { get; set; }
    }
}
