using System;
using System.Collections.Generic;
using static TorchSharp.torch;
using static TorchSharp.torch.utils;

namespace TorchLibrarys.BiLSTMCRF.Data
{
    public class BilstmDataloader : data.DataLoader<List<List<int>>, (Tensor, Tensor, Tensor, Tensor)>
    {
        //public Bilstm_dataloader(data.Dataset<List<List<int>>> dataset, int batchSize, Func<IEnumerable<List<List<int>>>, Device, (Tensor, Tensor, Tensor, Tensor)> collate_fn, IEnumerable<long> shuffler, Device device = null, int num_worker = 1) : base(dataset, batchSize, collate_fn, shuffler, device, num_worker)
        //{
        //}
        public BilstmDataloader(data.Dataset<List<List<int>>> dataset, int batchSize, Func<IEnumerable<List<List<int>>>, Device, (Tensor, Tensor, Tensor, Tensor)> collate_fn, bool shuffler, Device device = null, int num_worker = 1) : base(dataset, batchSize, collate_fn, shuffler, device, num_worker)
        {
        }
    }
}
