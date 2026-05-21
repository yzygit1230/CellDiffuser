import sys
from collections import OrderedDict
import data
from options.train_options import TrainOptions
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.cell_trainer import CellTrainer
from tqdm import tqdm

opt = TrainOptions().parse()
print(' '.join(sys.argv))

dataloader = data.create_dataloader(opt)

trainer = CellTrainer(opt)
iter_counter = IterationCounter(opt, len(dataloader))
visualizer = Visualizer(opt)

for epoch in tqdm(iter_counter.training_epochs()):
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(tqdm(dataloader), start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()
        trainer.run_generator_one_step(data_i)
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visuals = OrderedDict([('content', data_i['label'][0]),
                                   ('synthesized_image', trainer.get_latest_generated()[0]),
                                   ('style', data_i['image'][0])])
            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Train Finished.')
