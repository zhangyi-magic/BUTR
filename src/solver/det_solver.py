"""
Training and validation for BUTR and detector baselines.

Modified from Dome-DETR and D-FINE.
Copyright (c) 2025 The Dome-DETR Authors. All Rights Reserved.
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import datetime
import json
import time

import torch

from ..misc import dist_utils
from ..optim.flat_cosine import FlatCosineLRScheduler
from ._solver import BaseSolver
from .det_engine import evaluate, train_one_epoch


class DetSolver(BaseSolver):
    def fit(self):
        self.train()
        args = self.cfg

        if isinstance(self.cfg.train_dataloader.collate_fn.base_size, str):
            try:
                self.cfg.train_dataloader.collate_fn.base_size = int(
                    self.cfg.train_dataloader.collate_fn.base_size
                )
            except ValueError:
                try:
                    self.cfg.train_dataloader.collate_fn.base_size = (
                        self.cfg.train_dataloader.collate_fn.base_size.strip("()[]")
                    )
                    self.cfg.train_dataloader.collate_fn.base_size = tuple(
                        int(x.strip())
                        for x in self.cfg.train_dataloader.collate_fn.base_size.split(",")
                    )
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Cannot convert base_size string "
                        f"'{self.cfg.train_dataloader.collate_fn.base_size}' "
                        f"to valid size format"
                    )

        self.patience = 6
        self.not_improved_count = 0
        self.stage2_reload = "stage1"

        n_parameters = sum(p.numel() for p in self.model.parameters())
        print(f"Number of parameters: {n_parameters:,}")
        if args.yaml_cfg.get("profile_model", False):
            from ..misc.profiler_utils import stats

            _, model_stats = stats(self.cfg)
            print(model_stats)
        print("-" * 42 + "Start training" + "-" * 43)

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            if str(args.lrsheduler).lower() != "flatcosine":
                raise ValueError(f"Unsupported lrsheduler: {args.lrsheduler}")

            iter_per_epoch = len(self.train_dataloader)
            self.lr_scheduler = FlatCosineLRScheduler(
                self.optimizer,
                args.lr_gamma,
                iter_per_epoch,
                total_epochs=args.epoches,
                warmup_iter=args.warmup_iter,
                flat_epochs=args.flat_epoch,
                no_aug_epochs=args.no_aug_epoch,
            )
            self.self_lr_scheduler = True

        top1 = 0
        best_stat = {
            "epoch": -1,
        }

        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
            )
            for k in test_stats:
                best_stat["epoch"] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f"best_stat: {best_stat}")

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1

        val_start_epoch = 0

        for epoch in range(start_epoch, args.epoches):
            self.train_dataloader.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if self.output_dir and epoch == self.train_dataloader.collate_fn.stop_epoch:
                if dist_utils.is_dist_available_and_initialized():
                    torch.distributed.barrier()
                self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                if self.ema is not None:
                    self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                    print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                self_lr_scheduler=self.self_lr_scheduler,
                lr_scheduler=self.lr_scheduler,
            )

            if not self.self_lr_scheduler and (
                self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished()
            ):
                self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [self.output_dir / "last.pth"]
                # extra checkpoint before LR drop and every checkpoint_freq epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model

            should_validate = epoch >= val_start_epoch

            if should_validate:
                test_stats, coco_evaluator = evaluate(
                    module,
                    self.criterion,
                    self.postprocessor,
                    self.val_dataloader,
                    self.evaluator,
                    self.device,
                )

                for k in test_stats:
                    if self.writer and dist_utils.is_main_process():
                        for i, v in enumerate(test_stats[k]):
                            self.writer.add_scalar(f"Test/{k}_{i}".format(k), v, epoch)

                    if k in best_stat:
                        best_stat["epoch"] = (
                            epoch if test_stats[k][0] > best_stat[k] else best_stat["epoch"]
                        )
                        best_stat[k] = max(best_stat[k], test_stats[k][0])
                    else:
                        best_stat["epoch"] = epoch
                        best_stat[k] = test_stats[k][0]

                    if best_stat[k] > top1:
                        best_stat_print["epoch"] = epoch
                        self.not_improved_count = 0
                        top1 = best_stat[k]
                        if self.output_dir:
                            if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                                dist_utils.save_on_master(
                                    self.state_dict(), self.output_dir / "best_stg2.pth"
                                )
                            else:
                                dist_utils.save_on_master(
                                    self.state_dict(), self.output_dir / "best_stg1.pth"
                                )
                    else:
                        self.not_improved_count += 1

                    best_stat_print[k] = max(best_stat[k], top1)
                    print(f"current_stat: {test_stats[k][0]}")
                    print(f"best_stat: {best_stat_print}")  # global best

                    if best_stat["epoch"] == epoch and self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            if test_stats[k][0] > top1:
                                top1 = test_stats[k][0]
                                dist_utils.save_on_master(
                                    self.state_dict(), self.output_dir / "best_stg2.pth"
                                )
                        else:
                            top1 = max(test_stats[k][0], top1)
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg1.pth"
                            )

                    elif (
                        self.output_dir
                        and self.ema is not None
                        and epoch >= self.train_dataloader.collate_fn.stop_epoch
                    ):
                        if self.not_improved_count >= self.patience:
                            best_stat = {
                                "epoch": -1,
                            }
                            self.ema.decay -= 0.0001
                            self.not_improved_count = 0
                            if self.stage2_reload == "stage2":
                                self.load_resume_state(str(self.output_dir / "best_stg2.pth"))
                            else:
                                self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                            print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")
                        else:
                            print(
                                f"Tolerate undesirable result for patience: "
                                f"{self.not_improved_count} / {self.patience} "
                            )
            else:
                print(
                    f"Skip evaluation at epoch {epoch}. "
                    f"Validation starts after epoch {val_start_epoch}."
                )
                test_stats, coco_evaluator = {}, None

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
            if should_validate:
                log_stats.update({f"test_{k}": v for k, v in test_stats.items()})

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def val(
        self,
    ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
        )

        if self.output_dir and dist_utils.is_main_process():
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )

        return
