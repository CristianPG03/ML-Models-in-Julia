#First of all add the .jl with all the neccesary functions
println("Loading the functions...");
include("utils/FunctionsUnits.jl");
include("utils/LoadData.jl");
println("Functions Loaded.\n");


#Load the Packages that we will need and download it if neccesary
res = 0
while res != 1 && res != 2
    println("You want to download the packages? (1->y;2->n)")
    global res = parse(Int, readline())
    if res == 1
        println("Loading the Packages...");
        include("utils/DownloadPackages.jl");
        include("utils/LoadPackages.jl")
        println("Packages Loaded.\n");
    else 
        println("Loading the Packages...");
        include("utils/LoadPackages.jl")
        println("Packages Loaded.\n");
    end
end


#Load the data from the txt and printing the total size 
println("Loading dataset...");
data = OpenDataset();
println("The size of the entire dataset is: ", size(data), "\n");

#Initialize some variables that we will need later on
model_train = 1;
method = 1;
number = 1;
hidden_layers = 1;


#Doing a while in order to repeat the loop as many times as you want
while true
    println("Enter a approach: 1.- Approach 1, 2.- Approach 2, 3.- Approach 3, 4.- Approach 4, 5.- Exit")
    global number = parse(Int, readline())

    #APPROACH 1
    if number == 1 

        println("You select Approach 1\n");
        println("Transforming the data...")

        #Select the data and apply the normalization
        Random.seed!(123)
        local new_data = SelectYears(data, 5000);
        targets = new_data[:,1];
        inputs = Matrix{Float32}(new_data[:,2:91]);
        inputs = convert(Array{Float32, 2},normalizeZeroMean!(inputs));

        #Define the index for the crossvalidation proccess
        k = 10;
        Random.seed!(123);
        index = crossvalidation(targets, k); 

        #Print the new size of the data
        println("The size of the dataset for the Approach 1 is: ", size(inputs), "\n")
        global model_train = 1

        #Selecting if you want to train the data or load the trained models 
        while model_train == 1 || model_train == 2 

            println("Select: 1.-Train the models, 2.- Load the trained models, Other.- Return.")
            global model_train = parse(Int, readline())

            #Train the model 
            if model_train == 1
                println("You select: Train the models")
                include("utils/Train_models.jl")
                Train_models(inputs,targets,index)

            #Load the models 
            elseif model_train == 2
                println("You select: Load the models")
                include("utils/LoadModels.jl")
                LoadModels(number);

            end
        end

    #APPROACH 2 (Analogous to approach 1)
    elseif number == 2

        println("You select Approach 2");
        println("Transforming the data...")

        Random.seed!(123)
        local new_data = SelectYears(data, 7500);
        targets = new_data[:,1];
        inputs = Matrix{Float32}(new_data[:,2:13]);
        inputs = convert(Array{Float32, 2},normalizeZeroMean!(inputs));

        k = 10;
        Random.seed!(123);
        index = crossvalidation(targets, k); 


        println("The size of the dataset for the Approach 2 is: ", size(inputs), "\n")
        global model_train = 1


        while model_train == 1 || model_train == 2 

            println("Select: 1.-Train the models, 2.- Load the trained models, Other.- Return.")
            global model_train = parse(Int, readline())

            if model_train == 1
                println("You select: Train the models")
                include("utils/Train_models.jl")
                Train_models(inputs,targets,index)

            elseif model_train == 2
                println("You select: Load the models")
                include("utils/LoadModels.jl")
                LoadModels(number);

            end
        end


    #APPROACH 3 (Analogous to approach 1)
    elseif number == 3

        println("You select Approach 3");
        println("Transforming the data...")

        Random.seed!(123)
        local new_data = SelectYears(data, 5000);
        targets = new_data[:,1];
        inputs = Matrix{Float32}(new_data[:,14:91]);
        inputs = convert(Array{Float32, 2},normalizeZeroMean!(inputs));

        k = 10;
        Random.seed!(123);
        index = crossvalidation(targets, k); 


        println("The size of the dataset for the Approach 3 is: ", size(inputs), "\n")
        global model_train = 1


        while model_train == 1 || model_train == 2 

            println("Select: 1.-Train the models, 2.- Load the trained models, Other.- Return.")
            global model_train = parse(Int, readline())

            if model_train == 1
                println("You select: Train the models")
                include("utils/Train_models.jl")
                Train_models(inputs,targets,index)

            elseif model_train == 2
                println("You select: Load the models")
                include("utils/LoadModels.jl")
                LoadModels(number);

            end
        end


    # APPROACH 4, similar to approach 1, using Train_PCA
    elseif number == 4

        println("You select Approach 4");
        include("utils/Train_PCA.jl");
        println("Transforming the data...")

        Random.seed!(123)
        local new_data = SelectYears(data, 5000);
        targets = new_data[:,1];
        inputs = Matrix{Float32}(new_data[:,2:91]);
        inputs = convert(Array{Float32, 2},normalizeZeroMean!(inputs));

        k = 10;
        Random.seed!(123);
        index = crossvalidation(targets, k); 


        println("The size of the dataset for the Approach 4 is: ", size(inputs), "\n")
        global model_train = 1


        while model_train == 1 || model_train == 2 

            println("Select: 1.-Train the models, 2.- Load the trained models, Other.- Return.")
            global model_train = parse(Int, readline())

            if model_train == 1
                println("You select: Train the models")
                Train_models_PCA(inputs,targets,index)

            elseif model_train == 2
                println("You select: Load the models")
                include("utils/LoadModels.jl")
                LoadModels(number);

            end
        end

    else
        return

    end 
end