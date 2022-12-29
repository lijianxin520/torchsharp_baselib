using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace TorchLibrarys.BiLSTMCRF.Data
{
    /// <summary>
    /// 进行加载需要训练的文本
    /// </summary>
    public class DataProcessor
    {
        public string _data_dir { get; set; }
        public string[] _files { get; set; }
        public DataProcessor(string datadir,params string[] files)
        {
            _data_dir = datadir;
            _files = files;
        }

        public void data_process() 
        {
            //file_name = train && test
            foreach (var file_name in this._files)
            {
                this.get_examples(file_name);
            }
            
        }
        /// <summary>
        /// 处理样例数据
        /// </summary>
        /// <param name="mode"></param>
        /// <exception cref="Exception"></exception>
        private void get_examples(string mode)
        {
            /*
            函数功能：为样本数据集中的词组添加BMES标签，并将字的集合和标签的集合保存为二进制文件，方便之后模型训练读取

            参数 ：
                mode 样本数据集的函数名

            返回值：
                将txt文件每一行中的文本分离出来，存储为words列表，将所有行合并为word_list
                BMES标注法标记文本对应的标签，存储为labels，将所有行合并为 label_list
        
            细节：
                本函数中最重要的函数是第97行的getlist，其给每个字添加上了标签，具体细节详见该函数处说明
            **/
            
            string input_dir = Path.Combine(this._data_dir, mode + ".txt"); //输入文件 ,self.data_dir = os.getcwd() + '/data/'
            string output_dir = Path.Combine(this._data_dir,mode + ".npz");    //输出文件
            if(File.Exists(output_dir))
            {
                return;
            }
            var alllines= File.ReadAllLines(input_dir,encoding:Encoding.UTF8);
            List<List<char>> word_list = new List<List<char>>();
            List<List<char>> label_list = new List<List<char>>();
            int num = 0;
            foreach (var line in alllines)
            {
                //逐行读取文件    举例 "共同  创造  美好  的  新  世纪  ——  二○○一年  新年  贺词"
                num += 1;
                List<char> words = new List<char>();

                string linetemp = line.Trim(); // remove spaces at the beginning and the end
                                           //print(line)
                if (string.IsNullOrWhiteSpace(linetemp))
                {
                    //line is None
                    continue;
                }

                foreach (int i in Enumerable.Range(0, linetemp.Length))
                {
                    if (linetemp[i]==' ')
                    {
                        //skip space
                        continue;
                    }
                    //按字切分句子    words="共同创造美好的新世纪-—二○○一年新年贺词"
                    words.Add(linetemp[i]);
                }
                word_list.Add(words);             //word_list 字集
                var text = line.Trim().Split("  ");              //text=["共同","创造","美好","的","新","世纪","——","二○○一年","新年","贺词"]
                // print(text)
               List<char>  labels = new List<char>();
                foreach (var item in text)
                {
                    if (item == "") 
                    {
                        continue;
                    }
                    labels.AddRange(getlist(item));// 给训练集中的每行句子中的每个词语添加标签  举例 ： "二○○一年" 对应标签为 "BMMME"
                }
                label_list.Add(labels);  // label_list 标签集
                if (labels.Count()!= words.Count())
                {
                    throw new Exception("labels 数量与 words 不匹配");
                }
            }

            Console.WriteLine($"We have,{num}, lines in {mode},file processed");
            // 保存成二进制文件
            //这里先保存为json
            string data = JsonConvert.SerializeObject(new Dictionary<string, List<List<char>>> { { "words", word_list }, { "labels", label_list } });
            File.WriteAllText(output_dir, data, Encoding.UTF8);
            //np.savez_compressed(output_dir, words = word_list, labels = label_list);// 将word_list,label_list保存到一个二进制文件中
            /*
            np.savez_compressed 对应的读取二进制文件方法详见本网址 https://www.cnblogs.com/wushaogui/p/9142019.html
            **/
            Console.WriteLine("-------- {0} data process DONE!--------", mode);


        }
        /// <summary>
        /// 拆分内容
        /// </summary>
        /// <param name="input_str"></param>
        /// <returns></returns>
        private char[] getlist(string input_str) 
        {
            //将每个输入词转换为BMES标注
            List<char> output_str = new List<char>();
            switch (input_str.Length)
            {
                case 1:
                    output_str.Add('S');
                    break;
                case 2:
                    output_str.AddRange(new[] { 'B', 'E' });
                    break;
                default:
                   int M_num= input_str.Length - 2;
                    output_str.Add('B');
                    for (int i = 0; i < M_num; i++)
                    {
                        output_str.Add('M');
                    }
                    output_str.Add('E');
                    break;
            }
            return output_str.ToArray();
        }

    }
}
