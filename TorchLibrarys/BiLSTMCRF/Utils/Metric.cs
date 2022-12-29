using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp.Modules;
using static System.Formats.Asn1.AsnWriter;
using static TorchSharp.torch.distributions.constraints;

namespace TorchLibrarys.BiLSTMCRF.Utils
{
    public  class Metric
    {
        public static (float score, float p, float r) f1_score(List<List<char>> y_true, List<List<char>> y_pred) 
        {
            HashSet<(int,int)> true_entities = new HashSet<(int, int)>();
                get_entities(y_true).ForEach(a => true_entities.Add(a));

            HashSet<(int, int)> pred_entities = new HashSet<(int, int)>();
                get_entities(y_pred).ForEach(a => pred_entities.Add(a));
            //var true_entities = set(get_entities(y_true));
            //var pred_entities = set(get_entities(y_pred));
            //true_entities & pred_entities:取交集
            var nb_correct = true_entities.Intersect(pred_entities).Count();
            var nb_pred = pred_entities.Count;
            var nb_true = true_entities.Count;

            var p = nb_pred > 0 ? (float)nb_correct / nb_pred : 0;
            //var  p = nb_correct / nb_pred if nb_pred > 0 else 0;
            var r = nb_true > 0 ? (float)nb_correct / nb_true:0;
            //var r = nb_correct / nb_true if nb_true > 0 else 0;
            var score = (p + r) > 0 ? 2 * p * r / (p + r):0;
            //var score = 2 * p * r / (p + r) if p + r > 0 else 0;
            return (score, p, r);
        }

        public static List<(int, int)> get_entities(List<List<char>> seq) 
        {
            // for nested list
            //if any(isinstance(s, list) for s in seq):
            //  seq = [item for sublist in seq for item in sublist + ['O']]
            List<char> newseq=new List<char>();
            foreach (var sublist in seq)
            {
                var sublisttemp=new List<char>(sublist);
                sublisttemp.Add('O');
                foreach (var item in sublisttemp)
                {
                    newseq.Add(item);
                }
            }
            char prev_tag = 'O';
            int begin_offset = 0;
            List<(int, int)> chunks = new List<(int, int)>();
            //newseq.Select((item,i)=>(i+1,item))
            newseq.Add('O');
            for (int i = 0; i < newseq.Count; i++)
            {
                char chunk= newseq[i];
                // tag = chunk[0]
                char tag = chunk;
                if (end_of_chunk(prev_tag, tag))
                {
                    chunks.Add((begin_offset, i - 1));
                }

                if (start_of_chunk(prev_tag, tag)) 
                {
                    begin_offset = i;
                }
                prev_tag = tag;
            }
            return chunks;
        }

        private static bool end_of_chunk(char prev_tag, char tag)
        {
            /*Checks if a chunk ended between the previous and current word.
            Args:
                prev_tag: previous chunk tag.
                tag: current chunk tag.
            Returns:
                chunk_end: boolean.
            */
            bool chunk_end = false;

            if (prev_tag == 'S')
                chunk_end = true;
            if (prev_tag == 'E')
                chunk_end = true;
            // pred_label中可能出现这种情形
            if (prev_tag == 'B' && tag == 'B')
                chunk_end = true;
            if (prev_tag == 'B' && tag == 'S')
                chunk_end = true;
            if (prev_tag == 'B' && tag == 'O')
                chunk_end = true;
            if (prev_tag == 'M' && tag == 'B')
                chunk_end = true;
            if (prev_tag == 'M' && tag == 'S')
                chunk_end = true;
            if (prev_tag == 'M' && tag == 'O')
                chunk_end = true;

            return chunk_end;
        }


        private static bool start_of_chunk(char prev_tag, char tag)
        {
            /*Checks if a chunk started between the previous and current word.
            Args:
                prev_tag: previous chunk tag.
                tag: current chunk tag.
            Returns:
                chunk_start: boolean.
            */
            bool chunk_start = false;

            if (tag == 'B')
                chunk_start = true;
            if (tag == 'S')
                chunk_start = true;

            if (prev_tag == 'O' && tag == 'M')
                chunk_start = true;
            if (prev_tag == 'O' && tag == 'E')
                chunk_start = true;
            if (prev_tag == 'S' && tag == 'M')
                chunk_start = true;
            if (prev_tag == 'S' && tag == 'E')
                chunk_start = true;
            if (prev_tag == 'E' && tag == 'M')
                chunk_start = true;
            if (prev_tag == 'E' && tag == 'E')
                chunk_start = true;

            return chunk_start;
        }

        private static List<List<char>> calculate(List<char>  x, List<char> y)
        {
            /*
            Gets words of entities from sequence.
            Args:
                x (list): sequence of words.
                y (list): sequence of labels.
            Returns:
                res: list of entities.
        **/
            var res = new List<List<char>>();
            var entity = new List<char>();
            char prev_tag = 'O';  // start tag
            var yy = new List<char>();
            yy.Add('O');
            for (int i = 0; i < yy.Count; i++)
            {
                var tag = yy[i];
                if (end_of_chunk(prev_tag, tag))
                {
                    res.Add(entity);
                }
                if (start_of_chunk(prev_tag, tag) && i < x.Count)
                {
                    entity =new List<char>() { x[i] };
                } else if (i < x.Count)
                {
                    entity.Add(x[i]);
                }
                else 
                {
                    continue;
                }
                prev_tag = tag;
            }
            return res;
        }
    
    
        public static void output_write(string output_dir, List<List<char>> sents, List<List<char>> preds) 
        {
            //write results into output.txt for f1 calculation
            using (StreamWriter wt = new StreamWriter(output_dir, append: false, Encoding.UTF8))
            {
                foreach (var (s, p) in sents.Zip(preds))
                {
                    var res = calculate(s, p);
                    foreach (var entity in res)
                    {
                        foreach (var w in entity)
                        {
                            wt.Write(w);
                        }
                        wt.Write("  ");
                    }
                    wt.WriteLine("");
                }
            }
        }
        public static void bad_case(string case_dir, List<List<char>> sents, List<List<char>> preds, List<List<char>> tags)
        {
            //config.case_dir
            //string case_dir = "";
            if (!Directory.Exists(case_dir))
            {// 调用系统命令行来创建文件
                Directory.CreateDirectory(case_dir);
            }
            List<string> output=new List<string>();
            int idx = 0;
            foreach (var(t, p) in tags.Zip(preds)) 
            {
                if (t == p)
                    continue;
                else
                {
                    output.Add("bad case " + idx + ": ");
                    output.Add("sentence: " + string.Join(" ",sents[idx]));
                    output.Add("golden label: " + string.Join(" ", t ));
                    output.Add("model pred: " + string.Join(" ", p));
                }
            }
            File.AppendAllLines(case_dir, output, Encoding.UTF8);

            Console.WriteLine("--------Bad Cases reserved !--------");
        }
    }
}
