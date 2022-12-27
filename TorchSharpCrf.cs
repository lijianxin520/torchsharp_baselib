using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;
using static Tensorboard.CostGraphDef.Types;
using static Tensorboard.TensorShapeProto.Types;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace TorchBaseLib
{
    /// <summary>
    /// https://pytorch-crf.readthedocs.io/en/stable/_modules/torchcrf.html
    /// </summary>
    public class TorchSharpCrf : Module<Tensor,Tensor, Tensor, Tensor>
    {
        private long num_tags;
        private bool batch_first;
        private Parameter start_transitions;
        private Parameter end_transitions;
        private Parameter transitions;
        public TorchSharpCrf(long num_tags, bool batch_first=false, string name="crf") : base(name)
        {
            
            if (num_tags <= 0) 
            {
                throw new Exception($"invalid number of tags: {num_tags}");
            }
            this.num_tags = num_tags;
            this.batch_first = batch_first;
            this.start_transitions = nn.Parameter(torch.empty(num_tags));
            this.end_transitions = nn.Parameter(torch.empty(num_tags));
            this.transitions = nn.Parameter(torch.empty(num_tags, num_tags));

            this.Reset_parameters();

            //需要注册当前组件，才能获取到parameters()
            RegisterComponents();
        }

        /// <summary>
        /// Initialize the transition parameters.
        /// The parameters will be initialized randomly from a uniform distribution
        /// between -0.1 and 0.1.
        /// </summary>
        private void Reset_parameters() 
        {
            nn.init.uniform_(this.start_transitions, -0.1, 0.1);
            nn.init.uniform_(this.end_transitions, -0.1, 0.1);
            nn.init.uniform_(this.transitions, -0.1, 0.1);
        }
        public override Tensor forward(Tensor input1, Tensor input2, Tensor input3)
        {
           return this.Forward(input1, input2, input3);
        }
        private Tensor Forward(Tensor emissions, Tensor tags, Tensor? mask=null ,string reduction= "sum") 
        {
            this._validate(emissions, tags, mask);
            if(!new[] { "none", "sum", "mean", "token_mean" }.Contains(reduction))
            {
                throw new Exception($"invalid reduction: {reduction}");
            }

            mask ??= torch.ones_like(tags,ScalarType.Byte);

            if (this.batch_first) 
            {
                //张量类型转置，将小批量的数据放到最前面
                emissions = emissions.transpose(0, 1);
                tags = tags.transpose(0, 1);
                mask = mask.transpose(0, 1);
            }
            // shape: (batch_size,)
            Tensor numerator = this._compute_score(emissions, tags, mask);
            // shape: (batch_size,)
            Tensor denominator = this._compute_normalizer(emissions, mask);
            // shape: (batch_size,)
            var llh = numerator - denominator;

            if (reduction == "none")
                return llh;
            if (reduction == "sum")
                return llh.sum();
            if (reduction == "mean")
                return llh.mean();
            if (reduction != "token_mean")
                throw new Exception();
            return llh.sum() / mask.@float().sum();
        }
        /// <summary>
        /// Find the most likely tag sequence using Viterbi algorithm.
        /// </summary>
        /// <param name="emissions"></param>
        /// <param name="mask"></param>
        /// <returns></returns>
        public List<List<int>> Decode(Tensor emissions, Tensor? mask=null)
        {
            this._validate(emissions, mask: mask);
            if (mask is null)
            {   //mask = emissions.new_ones(emissions.shape[:2],ScalarType.Byte);
                //var emi = emissions.shape[..2];
                mask = emissions.new_ones(emissions.shape[0], emissions.shape[1], ScalarType.Byte);
                //mask = torch.ones(emi, ScalarType.Byte);
            }
            if (this.batch_first)
            {
                emissions = emissions.transpose(0, 1);
                mask = mask.transpose(0, 1);
            }
            return this._viterbi_decode(emissions, mask);
        }
        private void _validate(Tensor emissions, Tensor? tags=null, Tensor?  mask=null) 
        {
            if (emissions.dim() != 3) 
            {
                throw new Exception($"emissions must have dimension of 3, got {emissions.dim()}");
            }
            if (emissions.size(2) != this.num_tags)
            {
                throw new Exception($"expected last dimension of emissions is {this.num_tags},got {emissions.size(2)}");
            }
            if (!(tags is null) )
            {
                //数据比较对否一样，使用函数SequenceEqual；
                //bool a = emissions.shape[..2].SequenceEqual(tags.shape);
                if (!emissions.shape[..2].SequenceEqual(tags.shape))
                    throw new Exception($"the first two dimensions of emissions and tags must match, got {(emissions.shape[..2])} and {(tags.shape)}");
            }
            if (!(mask is null))
            {
                if (!emissions.shape[..2].SequenceEqual(mask.shape))
                    throw new Exception($"the first two dimensions of emissions and mask must match, got {(emissions.shape[..2])} and {(mask.shape)}");
                bool no_empty_seq = !this.batch_first && mask[0].all().ToBoolean();
                //var a=mask.select(1,0);//arg1:0：表示在行上选择，1：表示在列上选择；arg2：表示选择的下标；
                //bool no_empty_seq_bf = this.batch_first && a.all().ToBoolean();
                bool no_empty_seq_bf = this.batch_first && mask[1, 0].all().ToBoolean();
                if (!no_empty_seq && !no_empty_seq_bf)
                { 
                    throw new Exception("mask of the first timestep must all be on");
                }
            }

        }

        private Tensor _compute_score(Tensor emissions, Tensor tags, Tensor mask) 
        {
            //// emissions: (seq_length, batch_size, num_tags)
            //// tags: (seq_length, batch_size)
            //// mask: (seq_length, batch_size)
            //assert emissions.dim() == 3 && tags.dim() == 2;
            if (!(emissions.dim() == 3 && tags.dim() == 2))
                throw new Exception("");

            //assert emissions.shape[:2] == tags.shape;
            if (!(emissions.shape[..2].SequenceEqual(tags.shape)))
                throw new Exception("");
            //assert emissions.size(2) == this.num_tags;
            if (!(emissions.size(2) == this.num_tags))
                throw new Exception("");
            //assert mask.shape == tags.shape;
            if (!(mask.shape.SequenceEqual(tags.shape)))
                throw new Exception("");
            //assert mask[0].all();
            if (!(mask[0].all().ToBoolean()))
                throw new Exception("");
           
            int seq_length =Convert.ToInt32(tags.shape[0]);
            long batch_size = tags.shape[1];  
            mask = mask.@float();
            // Start transition score and first emission
            // shape: (batch_size,)
            var score = this.start_transitions[tags[0]];
            score += emissions[0, torch.arange(batch_size), tags[0]];
            foreach (var i in Enumerable.Range(1, seq_length-1))
            {
                // Transition score to next tag, only added if next timestep is valid (mask == 1)
                // shape: (batch_size,)
                score += this.transitions[tags[i - 1], tags[i]] * mask[i];


                // Emission score for next tag, only added if next timestep is valid (mask == 1)
                // shape: (batch_size,)
                score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i];
            }

            // End transition score
            // shape: (batch_size,)
            var seq_ends = mask.@long().sum(dim:0) - 1;
            // shape: (batch_size,)
            var last_tags = tags[seq_ends, torch.arange(batch_size)];
            // shape: (batch_size,)
            score += this.end_transitions[last_tags];
            return score;
        }

        private Tensor _compute_normalizer(Tensor emissions,Tensor mask) 
        {
            //// emissions: (seq_length, batch_size, num_tags)
            //// mask: (seq_length, batch_size)
            //assert emissions.dim() == 3 and mask.dim() == 2;
            //assert emissions.shape[:2] == mask.shape;
            //assert emissions.size(2) == self.num_tags;
            //assert mask[0].all();
            int seq_length =Convert.ToInt32(emissions.size(0));

            // Start transition score and first emission; score has size of
            // (batch_size, num_tags) where for each batch, the j-th column stores
            // the score that the first timestep has tag j
            // shape: (batch_size, num_tags)
           var score = this.start_transitions + emissions[0];
            foreach (var i in Enumerable.Range(1, seq_length-1))
            {
                // Broadcast score for every possible next tag
                // shape: (batch_size, num_tags, 1)
                var broadcast_score = score.unsqueeze(2);

                // Broadcast emission score for every possible current tag
                // shape: (batch_size, 1, num_tags)
                var broadcast_emissions = emissions[i].unsqueeze(1);
                // Compute the score tensor of size (batch_size, num_tags, num_tags) where
                // for each sample, entry at row i and column j stores the sum of scores of all
                // possible tag sequences so far that end with transitioning from tag i to tag j
                // and emitting
                // shape: (batch_size, num_tags, num_tags)
                var next_score = broadcast_score + this.transitions + broadcast_emissions;

                // Sum over all possible current tags, but we're in score space, so a sum
                // becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
                // all possible tag sequences so far, that end in tag i
                // shape: (batch_size, num_tags)
                next_score = torch.logsumexp(next_score, dim:1);

                // Set score to the next score if this timestep is valid (mask == 1)
                // shape: (batch_size, num_tags)
                //var condition= torch.tensor(mask[i].unsqueeze(1).data<Byte>().ToArray(), ScalarType.Bool).reshape(mask[i].unsqueeze(1).shape);
                score = torch.where(mask[i].unsqueeze(1).@bool(), next_score, score);
                //score = torch.where(mask[i].unsqueeze(1), next_score, score);

            }
            // End transition score
            // shape: (batch_size, num_tags)
            score += this.end_transitions;

            // Sum (log-sum-exp) over all possible tags
            // shape: (batch_size,)
            return torch.logsumexp(score, dim: 1);
        }



        private List<List<int>> _viterbi_decode(Tensor emissions, Tensor mask) 
        {
            //// emissions: (seq_length, batch_size, num_tags)
            //// mask: (seq_length, batch_size)
            //assert emissions.dim() == 3 and mask.dim() == 2;
            if (!(emissions.dim() == 3 && mask.dim() == 2))
                throw new Exception("");
            //assert emissions.shape[:2] == mask.shape;
            if (!(emissions.shape[..2].SequenceEqual(mask.shape)))
                throw new Exception("");
            //assert emissions.size(2) == self.num_tags;
            if (!(emissions.size(2) == this.num_tags))
                throw new Exception("");
            //assert mask[0].all();
            if (!(mask[0].all().ToBoolean()))
                throw new Exception("");
            int seq_length = (int)mask.shape[0];
            int batch_size =(int)mask.shape[1];


            // Start transition and first emission
            // shape: (batch_size, num_tags)
            var score = this.start_transitions + emissions[0];
            var history = new List<Tensor>();
            // score is a tensor of size (batch_size, num_tags) where for every batch,
            // value at column j stores the score of the best tag sequence so far that ends
            // with tag j
            // history saves where the best tags candidate transitioned from; this is used
            // when we trace back the best tag sequence
            // Viterbi algorithm recursive case: we compute the score of the best tag sequence
            // for every possible next tag
            foreach (var i in Enumerable.Range(1, seq_length-1))
            {
                // Broadcast viterbi score for every possible next tag
                // shape: (batch_size, num_tags, 1)
                var broadcast_score = score.unsqueeze(2);


                // Broadcast emission score for every possible current tag
                // shape: (batch_size, 1, num_tags)
                var broadcast_emission= emissions[i].unsqueeze(2);

                // Compute the score tensor of size (batch_size, num_tags, num_tags) where
                // for each sample, entry at row i and column j stores the score of the best
                // tag sequence so far that ends with transitioning from tag i to tag j and emitting
                // shape: (batch_size, num_tags, num_tags)
                var next_score = broadcast_score + this.transitions + broadcast_emission;


                // Find the maximum score over all possible current tag
                // shape: (batch_size, num_tags)
                var(next_score1, indices) = next_score.max(dim : 1);
                next_score = next_score1;

                // Set score to the next score if this timestep is valid (mask == 1)
                // and save the index that produces the next score
                // shape: (batch_size, num_tags)
                //var condition = torch.tensor(mask[i].unsqueeze(1).data<Byte>().ToArray(), ScalarType.Bool).reshape(mask[i].unsqueeze(1).shape);
                score = torch.where(mask[i].unsqueeze(1).@bool(), next_score, score);
                history.Add(indices);
            }

            // End transition score
            // shape: (batch_size, num_tags)
            score += this.end_transitions;

            // Now, compute the best path for each sample

            // shape: (batch_size,)
           var seq_ends = mask.@long().sum(dim: 0) - 1;
           var best_tags_list = new List<List<int>>();
            //Enumerable.Range 在.net中，参数1：表示其实位置；参数2：表示产生数量；
            //在py中的range，参数1：表示起始位置；参数2：表示停止位数，不包含后者；
            foreach (var idx in Enumerable.Range(0,batch_size))
            {
                // Find the tag which maximizes the score at the last timestep; this is our best tag
                // for the last timestep
                var ( _, best_last_tag) = score[idx].max(dim : 0);
                List<int> best_tags = new List<int>();
                best_tags.Add((int)best_last_tag.item<long>());
                // We trace back where the best last tag comes from, append that to our best tag
                // sequence, and trace it back again, and so on
                int seqidx =(int)seq_ends[idx].item<long>();
                var histroytemp= history.ToArray()[..seqidx];
                for (int i = seqidx - 1; i >-1; i--)
                {
                    var hist = histroytemp[i];
                    best_last_tag = hist[idx][best_tags[^1]];
                    best_tags.Add((int)best_last_tag.item<long>());
                }
                //Array.Reverse(histroytemp);
                //foreach (var hist in histroytemp)
                //{
                //    best_last_tag = hist[idx][best_tags[^1]];
                //    best_tags.Add((int)best_last_tag.item<long>());
                //}
                // Reverse the order because we start from the last timestep
                best_tags.Reverse();
                best_tags_list.Add(best_tags);
            }

            return best_tags_list;




        }

        
    }
}
