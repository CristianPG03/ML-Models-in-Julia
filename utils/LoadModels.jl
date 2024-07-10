#________________________________________________________________________________________________________________#
#________________________________________________________________________________________________________________#
#                               Function which load the trained models                                           #
#________________________________________________________________________________________________________________#
#________________________________________________________________________________________________________________#


#The function with a input that is "number" i.e. number of approach 
function LoadModels(number::Int64)
 
    method = 1
    
    #Select the method of the model that we are going to use
    while method == 1 || method == 2 || method == 3 || method == 4 || method == 5

        println("Select: 1.- ANN, 2.- SVM, 3.- kNN, 4.- Decision Tree, 5.- Ensemble Model, Other.- Return")
        global method = parse(Int, readline())

        # For the ANN models 
        if method == 1

            println("ANN");
            hidden_layers = 1

            #Select one hidden layer or two
            while hidden_layers == 1 || hidden_layers == 2
                println("Select: 1.- 1 Hidden Layer, 2.- 2 Hidden Layers, Other.- Return")
                global hidden_layers = parse(Int,readline())

                #ONE HIDDEN LAYER
                if hidden_layers == 1
                    #Name of the folders of all models
                    models_ann1 = ["modelann [15], [NNlib.softmax]","modelann [15], [tanh]", "modelann [15], [NNlib.sigmoid]",
                    "modelann [15], [NNlib.relu]", "modelann [30], [NNlib.relu]", "modelann [120], [NNlib.relu]",
                    "modelann [240], [NNlib.relu]", "modelann [480], [NNlib.relu]"];

                    n = length(models_ann1)
                    j = 0

                    #Initialize the vectors that will store the metrics of each model
                    mean_acc_mod_ann1 = zeros(n);
                    st_acc_mod_ann1 = zeros(n);
                    mean_f1_mod_ann1 = zeros(n);
                    st_f1_mod_ann1 = zeros(n);

                    #Iterate between all models
                    for model in models_ann1
                        j = j + 1
                        
                        #Initialize vectors that will store the metrics of each fold
                        vec_acc = zeros(10)
                        vec_f1 = zeros(10)

                        #Loop for open the targets and outputs for all folds
                        for i in 1:10

                            #Path to the targets
                            model_file_path = joinpath("utils", "Approach $number")
                            model_file_path = joinpath(model_file_path, model)
                            model_file_path = joinpath(model_file_path, "targets_annmodel$i.jld2")

                            #Load the targets
                            JLD2.jldopen(model_file_path) do file
                                global targets_model
                                targets_model = read(file, "targets_test")
                            end

                            #Path to the outputs
                            model2_file_path = joinpath("utils", "Approach $number")
                            model2_file_path = joinpath(model2_file_path, model)
                            model2_file_path = joinpath(model2_file_path, "outputs_annmodel$i.jld2")

                            #Load the targets
                            JLD2.jldopen(model2_file_path) do file
                                global outputs_model
                                outputs_model = read(file, "outputs_model")
                            end
                    
                            #Saving the metrics of the fold (Calling the function confusion matrix with the data loaded)
                            metrics = confusionMatrix(outputs_model, targets_model)
                            vec_acc[i] = metrics[12]
                            vec_f1[i] = metrics[6]

                        end

                        #Saving the metrics of each method
                        mean_acc_mod_ann1[j] = mean(vec_acc);
                        st_acc_mod_ann1[j] = std(vec_acc);
                        mean_f1_mod_ann1[j] = mean(vec_f1);
                        st_f1_mod_ann1[j] = std(vec_f1);

                    end

                    #Printing the metrics of each model
                    i = 1
                    for model in models_ann1
                        println("The model: ", model, " of the approach ", number, " have a mean accuracy: ", mean_acc_mod_ann1[i], " with a standard deviation of: ", st_acc_mod_ann1[i]);
                        println("and have an mean F1-score: ", mean_f1_mod_ann1[i], " with a standar deviation of: ", st_f1_mod_ann1[i]);
                        println(" ");
                        i += 1;
                    end 

                #Same process as with one hidden layer
                elseif hidden_layers == 2
                    models_ann2 = ["modelann [15, 15], [NNlib.sigmoid, NNlib.sigmoid]","modelann [15, 15], [tanh, tanh]","modelann [15, 15], [NNlib.softmax, NNlib.softmax]", 
                                    "modelann [15, 15], [NNlib.relu, NNlib.relu]", "modelann [120, 120], [NNlib.relu, NNlib.relu]", "modelann [240, 240], [NNlib.relu, NNlib.relu]",
                                    "modelann [480, 480], [NNlib.relu, NNlib.relu]", "modelann [240, 240], Function[NNlib.sigmoid, NNlib.relu]", "modelann [480, 240], Function[NNlib.sigmoid, NNlib.relu]"]
                    
                    n = length(models_ann2)
                    j = 0

                    mean_acc_mod_ann2 = zeros(n);
                    st_acc_mod_ann2 = zeros(n);
                    mean_f1_mod_ann2 = zeros(n);
                    st_f1_mod_ann2 = zeros(n);

                    for model in models_ann2
                        j = j + 1
                        vec_acc = zeros(10)
                        vec_f1 = zeros(10)

                        for i in 1:10

                            model_file_path = joinpath("utils", "Approach $number")
                            model_file_path = joinpath(model_file_path, model)
                            model_file_path = joinpath(model_file_path, "targets_annmodel$i.jld2")

                            JLD2.jldopen(model_file_path) do file
                                global targets_model
                                targets_model = read(file, "targets_test")
                            end

                            model2_file_path = joinpath("utils", "Approach $number")
                            model2_file_path = joinpath(model2_file_path, model)
                            model2_file_path = joinpath(model2_file_path, "outputs_annmodel$i.jld2")

                            JLD2.jldopen(model2_file_path) do file
                                global outputs_model
                                outputs_model = read(file, "outputs_model")
                            end
                    
                            metrics = confusionMatrix(outputs_model, targets_model)
                            vec_acc[i] = metrics[12]
                            vec_f1[i] = metrics[6]

                        end

                        mean_acc_mod_ann2[j] = mean(vec_acc);
                        st_acc_mod_ann2[j] = std(vec_acc);
                        mean_f1_mod_ann2[j] = mean(vec_f1);
                        st_f1_mod_ann2[j] = std(vec_f1);

                    end

                    i = 1
                    for model in models_ann2
                        println("The model: ", model, " of the approach ", number, " have a mean accuracy: ", mean_acc_mod_ann2[i], " with a standard deviation of: ", st_acc_mod_ann2[i]);
                        println("and have an mean F1-score: ", mean_f1_mod_ann2[i], " with a standar deviation of: ", st_f1_mod_ann2[i]);
                        println(" ");
                        i += 1;
                    end 

                
                else 
                    return
                end
 
            end
        

        #SVM
        elseif method == 2
            println("SVM")

            #APPROACH 1 (We need to diference approaches cause the models trained not are the same)
            #The rest of the code is Analogous to the first method
            if number == 1

                models_svm = ["modelsvm linear, 1, 2, 3","modelsvm poly, 1, 2, 2","modelsvm poly, 1, 2, 3","modelsvm poly, 1, 2, 5",
                                "modelsvm rbf, 1, 3, 3","modelsvm rbf, 1000, 20, 3","modelsvm sigmoid, 1, 2, 3","modelsvm sigmoid, 1000, 2, 3"]

                n = length(models_svm)
                j = 0

                mean_acc_mod_svm = zeros(n);
                st_acc_mod_svm = zeros(n);
                mean_f1_mod_svm = zeros(n);
                st_f1_mod_svm = zeros(n);

                for model in models_svm

                    j = j + 1
                    vec_acc = zeros(10)
                    vec_f1 = zeros(10)
                
                    for i in 1:10
                        
                        model_file_path = joinpath("utils", "Approach $number")
                        model_file_path = joinpath(model_file_path, model)
                        model_file_path = joinpath(model_file_path, "targets_svmmodel$i.jld2")

                        JLD2.jldopen(model_file_path) do file
                            global targets_model
                            targets_model = read(file, "targets_test")
                        end
                
                        model2_file_path = joinpath("utils", "Approach $number")
                        model2_file_path = joinpath(model2_file_path, model)
                        model2_file_path = joinpath(model2_file_path, "outputs_svmmodel$i.jld2")
                
                        JLD2.jldopen(model2_file_path) do file
                            global outputs_model
                            outputs_model = read(file, "outputs_model")
                        end
                
                        metrics = confusionMatrix(outputs_model, targets_model)
                        vec_acc[i] = metrics[12]
                        vec_f1[i] = metrics[6]
                
                    end
                
                    mean_acc_mod_svm[j] = mean(vec_acc);
                    st_acc_mod_svm[j] = std(vec_acc);
                    mean_f1_mod_svm[j] = mean(vec_f1);
                    st_f1_mod_svm[j] = std(vec_f1);

                end

                i = 1
                for model in models_svm
                    println("The model: ", model, " of the approach ", number," have a mean accuracy: ", mean_acc_mod_svm[i], " with a standard deviation of: ", st_acc_mod_svm[i]);
                    println("and have an mean F1-score: ", mean_f1_mod_svm[i], " with a standar deviation of: ", st_f1_mod_svm[i]);
                    println(" ");
                    i += 1;
                end 


            #APPROACH 2
            elseif number == 2

                models_svm = ["modelsvm linear, 1, 2, 3","modelsvm sigmoid, 1, 2, 3","modelsvm sigmoid, 500, 2, 3","modelsvm sigmoid, 1000, 2, 3",
                                "modelsvm rbf, 1, 3, 3","modelsvm rbf, 1, 20, 3","modelsvm rbf, 500, 10, 3","modelsvm rbf, 1000, 20, 3"]

                n = length(models_svm)
                j = 0

                mean_acc_mod_svm = zeros(n);
                st_acc_mod_svm = zeros(n);
                mean_f1_mod_svm = zeros(n);
                st_f1_mod_svm = zeros(n);

                for model in models_svm

                    j = j + 1
                    vec_acc = zeros(10)
                    vec_f1 = zeros(10)
                
                    for i in 1:10
                        
                        model_file_path = joinpath("utils", "Approach $number")
                        model_file_path = joinpath(model_file_path, model)
                        model_file_path = joinpath(model_file_path, "targets_svmmodel$i.jld2")

                        JLD2.jldopen(model_file_path) do file
                            global targets_model
                            targets_model = read(file, "targets_test")
                        end
                
                        model2_file_path = joinpath("utils", "Approach $number")
                        model2_file_path = joinpath(model2_file_path, model)
                        model2_file_path = joinpath(model2_file_path, "outputs_svmmodel$i.jld2")
                
                        JLD2.jldopen(model2_file_path) do file
                            global outputs_model
                            outputs_model = read(file, "outputs_model")
                        end
                
                        metrics = confusionMatrix(outputs_model, targets_model)
                        vec_acc[i] = metrics[12]
                        vec_f1[i] = metrics[6]
                
                    end
                
                    mean_acc_mod_svm[j] = mean(vec_acc);
                    st_acc_mod_svm[j] = std(vec_acc);
                    mean_f1_mod_svm[j] = mean(vec_f1);
                    st_f1_mod_svm[j] = std(vec_f1);

                end

                i = 1
                for model in models_svm
                    println("The model: ", model, " of the approach ", number, " have a mean accuracy: ", mean_acc_mod_svm[i], " with a standard deviation of: ", st_acc_mod_svm[i]);
                    println("and have an mean F1-score: ", mean_f1_mod_svm[i], " with a standar deviation of: ", st_f1_mod_svm[i]);
                    println(" ");
                    i += 1;
                end 

            #APPROACH 3 (Analogous)
            elseif number == 3

                models_svm = ["modelsvm linear, 1, 2, 3","modelsvm poly, 1, 2, 3","modelsvm rbf, 1, 3, 3","modelsvm rbf, 500, 10, 3","modelsvm rbf, 1000, 20, 3",
                                "modelsvm sigmoid, 1, 2, 3", "modelsvm sigmoid, 500, 2, 3","modelsvm sigmoid, 1000, 2, 3" ]

                n = length(models_svm)
                j = 0

                mean_acc_mod_svm = zeros(n);
                st_acc_mod_svm = zeros(n);
                mean_f1_mod_svm = zeros(n);
                st_f1_mod_svm = zeros(n);

                for model in models_svm

                    j = j + 1
                    vec_acc = zeros(10)
                    vec_f1 = zeros(10)
                
                    for i in 1:10
                        
                        model_file_path = joinpath("utils", "Approach $number")
                        model_file_path = joinpath(model_file_path, model)
                        model_file_path = joinpath(model_file_path, "targets_svmmodel$i.jld2")

                        JLD2.jldopen(model_file_path) do file
                            global targets_model
                            targets_model = read(file, "targets_test")
                        end
                
                        model2_file_path = joinpath("utils", "Approach $number")
                        model2_file_path = joinpath(model2_file_path, model)
                        model2_file_path = joinpath(model2_file_path, "outputs_svmmodel$i.jld2")
                
                        JLD2.jldopen(model2_file_path) do file
                            global outputs_model
                            outputs_model = read(file, "outputs_model")
                        end
                
                        metrics = confusionMatrix(outputs_model, targets_model)
                        vec_acc[i] = metrics[12]
                        vec_f1[i] = metrics[6]
                
                    end
                
                    mean_acc_mod_svm[j] = mean(vec_acc);
                    st_acc_mod_svm[j] = std(vec_acc);
                    mean_f1_mod_svm[j] = mean(vec_f1);
                    st_f1_mod_svm[j] = std(vec_f1);

                end

                i = 1
                for model in models_svm
                    println("The model: ", model, " of the approach ", number, " have a mean accuracy: ", mean_acc_mod_svm[i], " with a standard deviation of: ", st_acc_mod_svm[i]);
                    println("and have an mean F1-score: ", mean_f1_mod_svm[i], " with a standar deviation of: ", st_f1_mod_svm[i]);
                    println(" ");
                    i += 1;
                end 

            #APPROACH 4 (Analogous)
            elseif number == 4
                models_svm = ["modelsvm linear, 1, 2, 3","modelsvm poly, 1, 2, 3","modelsvm rbf, 1, 3, 3","modelsvm rbf, 500, 10, 5","modelsvm rbf, 1000, 20, 3",
                                "modelsvm sigmoid, 1, 2, 3","modelsvm sigmoid, 500, 2, 2","modelsvm sigmoid, 1000, 2, 3"]

                n = length(models_svm)
                j = 0
                mean_acc_mod_svm = zeros(n);
                st_acc_mod_svm = zeros(n);
                mean_f1_mod_svm = zeros(n);
                st_f1_mod_svm = zeros(n);
                for model in models_svm
                    j = j + 1
                    vec_acc = zeros(10)
                    vec_f1 = zeros(10)
                
                    for i in 1:10
                        
                        model_file_path = joinpath("utils", "Approach $number")
                        model_file_path = joinpath(model_file_path, model)
                        model_file_path = joinpath(model_file_path, "targets_svmmodel$i.jld2")

                        JLD2.jldopen(model_file_path) do file
                            global targets_model
                            targets_model = read(file, "targets_test")
                        end
                
                        model2_file_path = joinpath("utils", "Approach $number")
                        model2_file_path = joinpath(model2_file_path, model)
                        model2_file_path = joinpath(model2_file_path, "outputs_svmmodel$i.jld2")
                
                        JLD2.jldopen(model2_file_path) do file
                            global outputs_model
                            outputs_model = read(file, "outputs_model")
                        end
                
                        metrics = confusionMatrix(outputs_model, targets_model)
                        vec_acc[i] = metrics[12]
                        vec_f1[i] = metrics[6]
                
                    end
                
                    mean_acc_mod_svm[j] = mean(vec_acc);
                    st_acc_mod_svm[j] = std(vec_acc);
                    mean_f1_mod_svm[j] = mean(vec_f1);
                    st_f1_mod_svm[j] = std(vec_f1);

                end 
                
                i = 1
                for model in models_svm
                    println("The model: ", model, " of the approach ", number, " have a mean accuracy: ", mean_acc_mod_svm[i], " with a standard deviation of: ", st_acc_mod_svm[i]);
                    println("and have an mean F1-score: ", mean_f1_mod_svm[i], " with a standar deviation of: ", st_f1_mod_svm[i]);
                    println(" ");
                    i += 1;
                end 

            end 

        #kNN (Analogous to the rest of the methods)
        elseif method == 3
            println("kNN");

            models_knn = ["modelknn 1","modelknn 2","modelknn 3","modelknn 5","modelknn 10","modelknn 15"]

            n = length(models_knn)
            j = 0

            mean_acc_mod_knn = zeros(n); st_acc_mod_knn = zeros(n);
            mean_f1_mod_knn = zeros(n); st_f1_mod_knn = zeros(n);

            for model in models_knn

                j = j + 1
                vec_acc = zeros(10)
                vec_f1 = zeros(10)
            
                for i in 1:10
            
                    model_file_path = joinpath("utils", "Approach $number")
                    model_file_path = joinpath(model_file_path, model)
                    model_file_path = joinpath(model_file_path, "targets_knnmodel$i.jld2")
            
                    JLD2.jldopen(model_file_path) do file
                        global targets_model
                        targets_model = read(file, "targets_test")
                    end
            
                    model2_file_path = joinpath("utils", "Approach $number")
                    model2_file_path = joinpath(model2_file_path, model)
                    model2_file_path = joinpath(model2_file_path, "outputs_knnmodel$i.jld2")
            
                    JLD2.jldopen(model2_file_path) do file
                        global outputs_model
                        outputs_model = read(file, "outputs_model")
                    end
            
                    metrics = confusionMatrix(outputs_model, targets_model);
                    vec_acc[i] = metrics[12];
                    vec_f1[i] = metrics[6];

                end
            
                mean_acc_mod_knn[j] = mean(vec_acc); st_acc_mod_knn[j] = std(vec_acc);
                mean_f1_mod_knn[j] = mean(vec_f1); st_f1_mod_knn[j] = std(vec_f1);

            end

            i = 1
            for model in models_knn
                println("The model: ", model, " of the approach ", number, " have a mean accuracy: ", mean_acc_mod_knn[i], " with a standard deviation of: ", st_acc_mod_knn[i]);
                println("and have an mean F1-score: ", mean_f1_mod_knn[i], " with a standar deviation of: ", st_f1_mod_knn[i]);
                println(" ");
                i += 1;
            end 
      
        #Decission Tree (Analogous to the rest of the methods)
        elseif method == 4
            println("DT");

            models_dt = ["modeldt 5", "modeldt 20", "modeldt 30", "modeldt 40", "modeldt 50", "modeldt 100"]

            n = length(models_dt)
            j = 0

            mean_acc_mod_dt = zeros(n); st_acc_mod_dt = zeros(n);
            mean_f1_mod_dt = zeros(n); st_f1_mod_dt = zeros(n);

            for model in models_dt

                j = j + 1
                vec_acc = zeros(10)
                vec_f1 = zeros(10)
            
                for i in 1:10

                    model_file_path = joinpath("utils", "Approach $number")
                    model_file_path = joinpath(model_file_path, model)
                    model_file_path = joinpath(model_file_path, "targets_dtmodel$i.jld2")
            
                    JLD2.jldopen(model_file_path) do file
                        global targets_model
                        targets_model = read(file, "targets_test")
                    end
            
                    model2_file_path = joinpath("utils", "Approach $number")
                    model2_file_path = joinpath(model2_file_path, model)
                    model2_file_path = joinpath(model2_file_path, "outputs_dtmodel$i.jld2")
            
                    JLD2.jldopen(model2_file_path) do file
                        global outputs_model
                        outputs_model = read(file, "outputs_model")
                    end
            
                    metrics = confusionMatrix(outputs_model, targets_model);
                    vec_acc[i] = metrics[12];
                    vec_f1[i] = metrics[6];
                end
            
                mean_acc_mod_dt[j] = mean(vec_acc); st_acc_mod_dt[j] = std(vec_acc);
                mean_f1_mod_dt[j] = mean(vec_f1); st_f1_mod_dt[j] = std(vec_f1);

            end      
            
            i = 1
            for model in models_dt
                println("The model: ", model, " of the approach ", number, " have a mean accuracy: ", mean_acc_mod_dt[i], " with a standard deviation of: ", st_acc_mod_dt[i]);
                println("and have an mean F1-score: ", mean_f1_mod_dt[i], " with a standar deviation of: ", st_f1_mod_dt[i]);
                println(" ");
                i += 1;
            end 

        #Ensemble (Analogous to the rest of the menthods, only having one model not various so without the first loop)
        elseif method == 5
            println("Ensemble");

            model = "modelensemble"

            vec_acc = zeros(10)
            vec_f1 = zeros(10)
        
            for i in 1:10

                model_file_path = joinpath("utils", "Approach $number")
                model_file_path = joinpath(model_file_path, model)
                model_file_path = joinpath(model_file_path, "targets_ensemblemodel$i.jld2")
        
                JLD2.jldopen(model_file_path) do file
                    global targets_model
                    targets_model = read(file, "targets_test")
                end

                model2_file_path = joinpath("utils", "Approach $number")
                model2_file_path = joinpath(model2_file_path, model)
                model2_file_path = joinpath(model2_file_path, "outputs_ensemblemodel$i.jld2")
        
                JLD2.jldopen(model2_file_path) do file
                    global outputs_model
                    outputs_model = read(file, "outputs_model")
                end
        
                metrics = confusionMatrix(outputs_model, targets_model);
                vec_acc[i] = metrics[12];
                vec_f1[i] = metrics[6];
            end
        
            mean_acc_mod_ens = mean(vec_acc); st_acc_mod_ens = std(vec_acc);
            mean_f1_mod_ens = mean(vec_f1); st_f1_mod_ens = std(vec_f1);

            println("The ensemble model of the approach ", number, " have a mean accuracy: ", mean_acc_mod_ens, " with a standard deviation of: ", st_acc_mod_ens);
            println("and have an mean F1-score: ", mean_f1_mod_ens, " with a standar deviation of: ", st_f1_mod_ens);
            println(" ");
        
        else 
            return
        end
    end
end