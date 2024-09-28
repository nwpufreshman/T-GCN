import argparse
import traceback
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
import models
import tasks
import utils.callbacks
import utils.data
import utils.email
import utils.logging


DATA_PATHS = {
    "shenzhen": {"feat": "data/sz_speed.csv", "adj": "data/sz_adj.csv"},
    "losloop": {"feat": "data/los_speed.csv", "adj": "data/los_adj.csv"},
}


def get_model(args, dm):
    """
    模型选择函数
    根据命令行参数中的 model_name,返回相应的模型实例(GCN、GRU 或 TGCN),
    并传入相应的参数(例如邻接矩阵、输入维度、隐藏层维度)
    """
    model = None
    if args.model_name == "GCN":
        model = models.GCN(adj=dm.adj, input_dim=args.seq_len, output_dim=args.hidden_dim)
    if args.model_name == "GRU":
        model = models.GRU(input_dim=dm.adj.shape[0], hidden_dim=args.hidden_dim)
    if args.model_name == "TGCN":
        model = models.TGCN(adj=dm.adj, hidden_dim=args.hidden_dim)
    return model

def get_task(args, model, dm):
    """
    任务选择函数
    该函数根据 settings 参数，动态获取任务类(例如 "SupervisedForecastTask")，
    并初始化任务实例，将模型、特征最大值和其他参数传递给任务。
    """    
    task = getattr(tasks, args.settings.capitalize() + "ForecastTask")(
        model=model, feat_max_val=dm.feat_max_val, **vars(args)
    )
    return task


def get_callbacks(args):
    """
    回调函数
    初始化了两个回调函数：一个是用于保存训练过程中模型的 ModelCheckpoint,
    另一个是用于在验证过程中绘制预测结果的自定义回调 PlotValidationPredictionsCallback。
    """
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="train_loss")
    plot_validation_predictions_callback = utils.callbacks.PlotValidationPredictionsCallback(monitor="train_loss")
    callbacks = [
        checkpoint_callback,
        plot_validation_predictions_callback,
    ]
    return callbacks


def main_supervised(args):
    """
    监督学习主函数
    该函数是代码的核心训练逻辑。首先，它通过 SpatioTemporalCSVDataModule 加载时空数据，
    然后获取模型、任务和回调函数，并初始化一个 PyTorch Lightning 的 Trainer 实例。
    训练通过 trainer.fit() 执行，训练完成后通过 trainer.validate() 进行验证，并返回结果。
    """
    dm = utils.data.SpatioTemporalCSVDataModule(
        feat_path=DATA_PATHS[args.data]["feat"], adj_path=DATA_PATHS[args.data]["adj"], **vars(args)
    )
    model = get_model(args, dm)
    task = get_task(args, model, dm)
    callbacks = get_callbacks(args)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer.fit(task, dm)
    results = trainer.validate(datamodule=dm)
    return results


def main(args):
    rank_zero_info(vars(args))
    results = globals()["main_" + args.settings](args)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser() # 使用 argparse 库创建一个新的命令行参数解析器对象 parser。
    parser = pl.Trainer.add_argparse_args(parser) # 将 PyTorch Lightning 的 Trainer 类中预定义的参数添加到我们刚刚创建的parser对象中

    # 调用 parser.add_argument() 来向解析器添加命令行参数。
    parser.add_argument(
        "--data", type=str, help="The name of the dataset", choices=("shenzhen", "losloop"), default="losloop"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model for spatiotemporal prediction",
        choices=("GCN", "GRU", "TGCN"),
        default="TGCN",
    )
    parser.add_argument(
        "--settings",
        type=str,
        help="The type of settings, e.g. supervised learning",
        choices=("supervised",),
        default="supervised",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to the output console log file")
    parser.add_argument("--send_email", "--email", action="store_true", help="Send email when finished")

    temp_args, _ = parser.parse_known_args() # 用于解析已知的命令行参数并忽略未知的参数。

    parser = getattr(utils.data, temp_args.settings.capitalize() + "DataModule").add_data_specific_arguments(parser)
    parser = getattr(models, temp_args.model_name).add_model_specific_arguments(parser)
    parser = getattr(tasks, temp_args.settings.capitalize() + "ForecastTask").add_task_specific_arguments(parser)
    parser.max_epochs = 100 

    # 这行代码执行最终的参数解析。此时，所有通过动态添加的参数都会被解析。
    args = parser.parse_args()
    utils.logging.format_logger(pl._logger)
    if args.log_path is not None:
        utils.logging.output_logger_to_file(pl._logger, args.log_path)

    try:
        results = main(args)
    except:  # noqa: E722
        traceback.print_exc()
        # if args.send_email:
        #     tb = traceback.format_exc()
        #     subject = "[Email Bot][❌] " + "-".join([args.settings, args.model_name, args.data])
        #     utils.email.send_email(tb, subject)
        exit(-1)

    # if args.send_email:
    #     subject = "[Email Bot][✅] " + "-".join([args.settings, args.model_name, args.data])
    #     utils.email.send_experiment_results_email(args, results, subject=subject)
