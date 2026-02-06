import os
import math
from decimal import Decimal

import utils
import wandb_utils

import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader['train']
        self.loader_test = loader['val']
        self.model = my_model
        self.loss = my_loss
        self.cls_loss = nn.CrossEntropyLoss()
        self.optimizer = utils.make_optimizer(args, self.model)
        self.scheduler = utils.make_scheduler(args, self.optimizer)

        # Initialize WandB logger (safely)
        self.wandb_logger = None
        if hasattr(args, 'use_wandb') and args.use_wandb:
            try:
                self.wandb_logger = wandb_utils.WandbLogger(args, my_model, my_loss)
            except Exception as e:
                print(f"⚠️ Failed to initialize WandB logger: {e}")
                print("Continuing without WandB...")

        # Training step counter for step-based checkpoint saving
        self.global_step = 0

        if self.args.resume == 1:
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        learn_rate = self.scheduler.get_last_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(learn_rate))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utils.timer(), utils.timer()

        for batch, (lr, hr, file_names) in enumerate(self.loader_train):
            # Update global step counter for checkpoint saving
            self.global_step += 1

            lr, hr = self.prepare([lr, hr])

            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr)

            loss = self.loss(sr, hr)

            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.01)
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.scheduler.step()
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()
        crop_border = self.scale[0]
        timer_test = utils.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                # self.loader_test.dataset.set_scale(idx_scale)
                # tqdm_test = tqdm(self.loader_test, ncols=80)

                img_num = 0
                for idx_img, (lr, hr, file_names) in enumerate(self.loader_test):
                    filename = file_names[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare([lr, hr])
                    else:
                        lr = self.prepare([lr])[0]

                    if self.args.test_block:
                        # test block-by-block

                        b, c, h, w = lr.shape
                        factor = self.scale[0] if not self.args.cubic_input else 1
                        tp = self.args.patch_size
                        if not self.args.cubic_input:
                            ip = tp // factor
                        else:
                            ip = tp

                        assert h >= ip and w >= ip, 'LR input must be larger than the training inputs'
                        if not self.args.cubic_input:
                            sr = torch.zeros((b, c, h * factor, w * factor))
                        else:
                            sr = torch.zeros((b, c, h, w))

                        for iy in range(0, h, ip):

                            if iy + ip > h:
                                iy = h - ip
                            ty = factor * iy

                            for ix in range(0, w, ip):

                                if ix + ip > w:
                                    ix = w - ip
                                tx = factor * ix

                                # forward-pass
                                lr_p = lr[:, :, iy:iy+ip, ix:ix+ip]
                                sr_p = self.model(lr_p)
                                sr[:, :, ty:ty+tp, tx:tx+tp] = sr_p

                    else:
                        sr = self.model(lr)
                    sr = utils.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    if not no_eval:

                        sr = utils.torch_to_np(sr)
                        hr = utils.torch_to_np(hr)
                        if self.args.test_y:
                            # first turn rgb into bgr
                            sr = sr[0, :, :, ::-1]
                            hr = hr[0, :, :, ::-1]
                            sr = utils.bgr2ycbcr(sr)
                            hr = utils.bgr2ycbcr(hr)

                        # crop borders
                        if crop_border == 0:
                            cropped_hr = hr
                            cropped_sr = sr
                        else:
                            cropped_hr = hr[:, crop_border:-crop_border, crop_border:-crop_border, :]
                            cropped_sr = sr[:, crop_border:-crop_border, crop_border:-crop_border, :]

                        if self.args.test_metric == 'psnr':
                            eval_acc += utils.calculate_psnr(cropped_sr, cropped_hr, self.args.rgb_range)
                        elif self.args.test_metric == 'ssim':
                            if self.args.rgb_range == 1:
                                eval_acc += utils.calculate_batch_ssim(cropped_sr * 255, cropped_hr * 255)
                            else:
                                eval_acc += utils.calculate_batch_ssim(cropped_sr, cropped_hr)
                        else:
                            print("No support this evaluation")

                        save_list.extend([lr, hr])
                        img_num += sr.shape[0]

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_acc / img_num
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\t{}: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.dataset,
                        scale,
                        self.args.test_metric,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

                # Log validation metrics to WandB (only for final scale, epoch-level logging)
                print(f"DEBUG: WandB check - hasattr: {hasattr(self, 'wandb_logger')}, not None: {self.wandb_logger is not None}, idx_scale: {idx_scale}")
                if hasattr(self, 'wandb_logger') and self.wandb_logger is not None and idx_scale == 0:
                    current_psnr = self.ckp.log[-1, idx_scale].item()
                    best_psnr = best[0][idx_scale].item()
                    print(f"DEBUG: Logging to WandB - epoch: {epoch}, psnr: {current_psnr}")

                    self.wandb_logger.log_validation(
                        epoch=epoch,
                        psnr_value=current_psnr,
                        ssim_value=None,  # 可以后续添加
                        best_psnr=best_psnr
                    )
                    print(f"DEBUG: WandB log_validation called successfully")
                else:
                    print(f"DEBUG: WandB logging skipped")

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc())
        )

        # Enhanced checkpoint saving
        if not self.args.test_only:
            # Regular epoch-based saving
            is_best = best[1][0] + 1 == epoch
            self.ckp.save(self, epoch, is_best=is_best)

            # Step-based checkpoint saving for cosine annealing
            if hasattr(self.args, 'save_every_n_steps') and self.args.save_every_n_steps > 0:
                if self.global_step % self.args.save_every_n_steps == 0:
                    # Save step-based checkpoint
                    checkpoint_path = os.path.join(self.ckp.dir, f'checkpoint_step_{self.global_step}.pt')
                    torch.save({
                        'global_step': self.global_step,
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'loss': self.loss.log[-1] if hasattr(self.loss, 'log') else None,
                    }, checkpoint_path)
                    print(f"💾 Saved step checkpoint at step {self.global_step}")

            # Model artifacts are saved separately by the training loop

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            # Finish WandB if in test mode
            if hasattr(self, 'wandb_logger'):
                self.wandb_logger.finish()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            finished = epoch >= self.args.epochs
            if finished and hasattr(self, 'wandb_logger'):
                # Finish WandB when training is complete
                self.wandb_logger.finish()
            return finished

