#!/usr/bin/env julia

function run_minibude(ppwi::Int)
    start_time = time()
    cmd = `julia --project=. src/JACCBUDE.jl -p $ppwi --deck src/data/bm2`
    
    # Create a unique log file for each run
    log_file = "minibude_p$(ppwi).log"
    
    println("Running miniBUDE with $ppwi ppwi")
    println("Logging output to $log_file")
    
    # Run the command and redirect output to log file
    open(log_file, "w") do io
        process = run(pipeline(cmd, stdout=io, stderr=io), wait=false)
        pid = getpid(process)
        
        while process_running(process)
            if (time() - start_time) > 150
                try
                    run(`kill -9 $pid`)
                catch e
                    println("Error killing process $pid: $e")
                end
                
                println("Timeout reached for p=$ppwi, deleting log file...")
                close(io)
                rm(log_file)
                return
            end
            sleep(0.1)
        end
    end
    
    # Small delay after completion
    sleep(5)
end

function main()
    ppwi = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Create results directory if it doesn't exist
    mkpath("minibude_results")
    
    # Track start time
    start_time = time()
    
    for p in ppwi
        try
            run_minibude(p)
            println("Completed run with $p ppwi")
        catch e
            println("Error running with $p ppwi: $e")
        end
    end
    
    # Calculate and print total runtime
    total_time = time() - start_time
    println("\nAll runs completed in $(round(total_time/60, digits=2)) minutes")
    
end

# Run the main function
main()
