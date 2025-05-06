'''
by lyuwenyu
'''
import time 
import json
import datetime
from pathlib import Path

import torch 
import wandb

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):
    
    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg 
        
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, }

        yaml_cfg = args.yaml_cfg
        if not yaml_cfg['wandb_name']:
            raise Exception("please config wandb_name")

        if not yaml_cfg['wandb_id']:
            raise Exception("please config wandb_id")

        # Initialize wandb
        if dist.is_main_process() and yaml_cfg['use_wandb']:
            wandb.init(
                project="deformable_detr",  # 项目名称
                name=yaml_cfg['wandb_name'],  # 实验名称
                id=yaml_cfg['wandb_id'],
                config=args,  # 记录配置参数
                dir=str(self.output_dir),  # 日志保存目录
                resume='auto'
            )
            # 记录模型结构
            wandb.watch(self.model, log="all", log_freq=100)

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler, 
                use_wandb=dist.is_main_process())

            self.lr_scheduler.step()
            
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, 
                self.device, self.output_dir, epoch=epoch, use_wandb=dist.is_main_process()
            )

            # 更新最佳状态
            should_save = False
            for k in test_stats.keys():
                if k in best_stat:
                    if test_stats[k][0] > best_stat[k]:
                        best_stat[k] = test_stats[k][0]
                        best_stat['epoch'] = epoch
                        should_save = True
                else:
                    best_stat[k] = test_stats[k][0]
                    best_stat['epoch'] = epoch
                    should_save = True
            
            # 只在性能提升时保存checkpoint
            if should_save and self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                # 额外保存一个带epoch编号的checkpoint
                # checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)
                    
                # 保存最佳模型到wandb
                # if dist.is_main_process():
                #     wandb.save(str(checkpoint_paths[0]))
            
            print('best_stat: ', best_stat)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        
        # Finish wandb run
        if dist.is_main_process():
            wandb.finish()


    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        # Initialize wandb for validation
        yaml_cfg = self.cfg.yaml_cfg
        if dist.is_main_process() and yaml_cfg['use_wandb']:
            wandb.init(
                project="deformable_detr_test",
                name=f"test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.cfg,
                dir=str(self.output_dir),
            )
            
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module, self.criterion, self.postprocessor,
            self.val_dataloader, base_ds, self.device, self.output_dir,
            use_wandb=dist.is_main_process()
        )
                
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        # Finish wandb run
        if dist.is_main_process():
            wandb.finish()
            
        return
