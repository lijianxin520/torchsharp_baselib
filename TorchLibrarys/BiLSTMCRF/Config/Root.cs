using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TorchLibrarys.BiLSTMCRF.Config
{
    public class Root
    {
        /// <summary>
        /// 
        /// </summary>
        public Wordseg wordseg { get; set; }
    }
    public class Wordseg
    {
        /// <summary>
        /// 
        /// </summary>
        public string data_dir { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public string train_dir { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public string test_dir { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public List<string> files { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public string vocab_path { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public string exp_dir { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public string model_dir { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public string log_dir { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public string case_dir { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public string output_dir { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public int max_vocab_size { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public int n_split { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public double dev_split_size { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public int batch_size { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public int embedding_size { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public int hidden_size { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public int lstm_layers { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public double lstm_drop_out { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public int nn_drop_out { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public double lr { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public List<double> betas { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public int lr_step { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public double lr_gamma { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public int epoch_num { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public int min_epoch_num { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public double patience { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public int patience_num { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public string gpu { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public Dictionary<char,int> label2id { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public Dictionary<int, char> id2label { get; set; }
    }
}
