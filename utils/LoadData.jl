#________________________________________________________________________________________________________________#
#________________________________________________________________________________________________________________#
#                                  Functions for load and preprocess the data                                    #
#________________________________________________________________________________________________________________#
#________________________________________________________________________________________________________________#



#Function for open the dataset and select the songs between 1995 and 2010
function OpenDataset()
    data_or = readdlm("datasets/YearPredictionMSD.txt", ',');
    println("Dataset loaded.\n")
    targets = data_or[:,1];
    targets = round.(Int,targets);
    index = ((targets .> 1995) .& (targets .< 2011))
    data = data_or[index,:];
    return data
end


#Function for select nsongs per year (compleatly random)
function SelectYears(data::AbstractArray{<:Any,2}, nsongs::Int64)
    unique_years = unique(data[:,1]);
    global new_data = [];
    for year in unique_years
        year_data = data[(data[:,1] .== year), :];
        local index = collect(1:size(year_data,1));
        index_reduced = StatsBase.sample(1:size(year_data,1), nsongs, replace=false)
        year_data = data[index_reduced, :];
        global new_data = vcat(new_data, year_data)
    end
    return new_data
end