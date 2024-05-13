import torch.utils.data.dataloader
from data_utils import *
from loss_utils import *
from model_utils import *
from metrics_utils import*
from configUnet3 import config_func_unet3
import argparse
import logging
# from PIL import Image
# from torchinfo import summary

#Printing in format defined by Katib yaml so Katib can parse the output
def end_of_epoch_print(epoch,IOU):
    ispis = "{{metricName: accuracy, metricValue: {:.4f}}}\n".format(IOU)
    print(ispis)
    logging.info(ispis)

def main(putanja_train, putanja_val, putanja_test, p_index,lr,lambda_p,step, num_epochs, loss_type,Batch_size, cfg):

    #Setting the variables
    lr = lr
    lambda_parametri = lambda_p
    stepovi = step
    epochs = num_epochs
    batch_size = Batch_size

    #Loading datasets
    train_loader, valid_loader = data_loading(putanja_train, putanja_val, cfg.binary, p_index, cfg.net_type, batch_size)
    test_loader = AgroVisionDataLoader(cfg.img_size, putanja_test, cfg.img_data_format, p_index,batch_size, 
                                       cfg.device, cfg.zscore, cfg.binary, cfg.dataset)
    #Model and loss init
    segmentation_net = model_init(cfg.num_channels, cfg.num_channels_lab, cfg.img_h, cfg.img_w, cfg.zscore,
                                cfg.net_type, cfg.device, cfg.server, cfg.GPU_list)
    segmentation_net = torch.nn.DataParallel(segmentation_net, device_ids=[0]).to(cfg.device)

    optimizer, scheduler = optimizer_init(segmentation_net, lr, cfg.weight_decay, cfg.scheduler_lr, lambda_parametri, cfg.optimizer_patience)
    criterion = loss_init(cfg.use_weights, loss_type, cfg.dataset, cfg.num_channels_lab, cfg.device, cfg.use_mask)

    # Counting Iterations
    epoch_list = np.zeros([epochs])
    all_train_losses = np.zeros([epochs])
    all_validation_losses = np.zeros([epochs])

    # print(summary(segmentation_net, (2,5,512,512)))
    
    if cfg.server:
            torch.cuda.empty_cache()
    for epoch in range(epochs):
        
        #Training
        train_part = "Train"
        segmentation_net.train(mode=True)

        for input_var, target_var, batch_names_train in train_loader:
            if cfg.server:
                print("Converting input to " + device)
                input_var, target_var = input_var.to(device), target_var.to(device)
            set_zero_grad(segmentation_net)

            model_output = segmentation_net.forward(input_var)
            print("Training: ", end="")
            loss = loss_calc(loss_type, criterion, model_output, target_var, cfg.num_channels_lab, cfg.use_mask)
            loss.backward()
            print(loss)
            optimizer.step()  # mnozi sa grad i menja weightove
            cfg.train_losses.append(loss.data)
            cfg.count_train += 1

        all_train_losses[epoch] = (torch.mean(torch.tensor(cfg.train_losses,dtype = torch.float32)))
        if epoch !=0 and (epoch % stepovi)==0 :
            print("epoha: " + str(epoch) +" , uradjen step!")

        if cfg.server:
            torch.cuda.empty_cache()


        #Validation
        train_part = "Valid"
        # segmentation_net.eval()
        with torch.no_grad():

            valid_scores = []
            
            for input_var, target_var, batch_names_valid in valid_loader:
                if cfg.server:
                    print("Converting input to " + device)
                    input_var, target_var = input_var.to(device), target_var.to(device)
                model_output = segmentation_net.forward(input_var)
                val_loss = loss_calc(loss_type,criterion,model_output,target_var, cfg.num_channels_lab ,cfg.use_mask)
                print("Validation: ", end="")
                print(val_loss)
                cfg.validation_losses.append(val_loss.data)
                cfg.count_val += 1
                #Racunanje F1 score-a na batchu - triple (accuracy, recall, batch_len)
                batch_results = calc_f1_batch(model_output, target_var, cfg.server)
                for score in batch_results:
                    valid_scores.append(score)

            

        scheduler.step(torch.mean(torch.tensor(cfg.validation_losses)))
        ##############################################################
        ### Racunanje finalne metrike nad celim validacionim setom ###
        ##############################################################
        sum_accuracy = 0
        sum_recall = 0
        print(len(valid_scores))
        for f1_score in valid_scores:
            print(f1_score)
            sum_accuracy += f1_score[0]
            sum_recall += f1_score[1]
        sum_accuracy = sum_accuracy / len(valid_scores)
        sum_recall = sum_recall / len(valid_scores)
        print("FINAL RESULTS")
        print(sum_accuracy, sum_recall)
        final_valid_metric = 2 * ((sum_accuracy * sum_recall)/(sum_accuracy + sum_recall))
        epoch_list[epoch] = epoch
        all_validation_losses[epoch] = (torch.mean(torch.tensor(cfg.validation_losses,dtype = torch.float32)))
        end_of_epoch_print(epoch,final_valid_metric)

    ######## TEST ############
    # CREATING IMAGES TO SEE IF NETWORKS ARE OUTPUTTING JUNK OR NOT
    # i = 0
    # for input_var, target_var, batch_names_valid in test_loader:
    #     i+=1
    #     model_output = segmentation_net.forward(input_var)
    #     sigmoid_function = torch.nn.Sigmoid()
    #     model_output = sigmoid_function(model_output)
        
    #     #extracting single images from batches
    #     for x in range(model_output.shape[0]):
            
    #         #only drawing pixels with value above 0.5
    #         thresholded = model_output[x, :, :, :]>0.5
    #         thresholded_tmp = thresholded.byte()
    #         picturex = torch.squeeze(thresholded_tmp)
    #         picturex = picturex.detach().numpy()
    #         #converting to float so picture can be drawn in png format
    #         picturex = picturex.astype(float)
    #         im = Image.fromarray(picturex, 'L')
    #         im.save("Slike/" +cfg.net_type + "/" + cfg.net_type + str(i) + str(x) + ".png")

            
    #         #drawing the target images for comparison
    #         thresholded_target = target_var[x, :, :, :]>0.5
    #         thresholded_target_tmp = thresholded_target.byte()
    #         picturex_target = torch.squeeze(thresholded_target_tmp)
    #         picturex_target = picturex_target.detach().numpy()
    #         picturex_target = picturex_target.astype(float)
    #         im_target = Image.fromarray(picturex_target, 'L')
    #         im_target.save("Slike/" + cfg.net_type + "/Test_" + cfg.net_type + str(i) + str(x) + ".png")

if __name__ == '__main__':
    print(torch.__version__)
    print("CUDA AVAILABILITY " + str(torch.cuda.is_available()))
    print(torch.version.cuda)
    parser = argparse.ArgumentParser(description='My program description')
    parser.add_argument('--learning_rate', type=float, default=0.001, metavar="N")
    parser.add_argument('--lambda_parametar', type=int, default=0.99, metavar="N")
    parser.add_argument('--stepovi_arr', type=int, default=5, metavar="N")
    parser.add_argument('--num_epochs', type=int, default=1, metavar="N")
    parser.add_argument('--loss_type', type=str, default="bce", metavar="N")
    parser.add_argument('--Batch_size', type=int, default=2, metavar="N")
    parser.add_argument('--net_type', type=str, default="UNet3", metavar="N")
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--trening_location', default="FullSet/trening_set_mini2/img", type=str)
    parser.add_argument('--validation_location', default="FullSet/validation_set_mini2/img", type=str)
    parser.add_argument('--test_location', default="FullSet/test_set_mini2/img", type=str)
    parser.add_argument('--new_location', default="", type=str)
    args = parser.parse_args()

    device = args.device
    if device == "cuda":
        config = config_func_unet3(True, args.net_type)
    else:
        config = config_func_unet3(False, args.net_type)
    data_location = args.new_location

    logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
            level=logging.DEBUG,
            filename=data_location)

    learning_rate = args.learning_rate
    lambda_parametar = args.lambda_parametar
    stepovi_arr = args.stepovi_arr
    num_epochs = args.num_epochs
    loss_type = args.loss_type
    net_type = args.net_type
    Batch_size = args.Batch_size

    trening_location = args.trening_location
    validation_location = args.validation_location
    test_location = args.test_location

    print("---inputs----")
    print(learning_rate)
    print(lambda_parametar)
    print(stepovi_arr)
    print(num_epochs)
    print(loss_type)
    print(Batch_size)
    print(net_type)
    print(config.device)
    print(trening_location)
    print(validation_location)
    print(test_location)
    print("-----------------")
    param_ponovljivosti = 1

    main(trening_location, validation_location, test_location, param_ponovljivosti, learning_rate, lambda_parametar, stepovi_arr,
                 num_epochs, loss_type, Batch_size, config)
    print("End of training component")
