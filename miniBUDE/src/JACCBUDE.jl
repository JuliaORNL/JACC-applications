# IMPORTANT NOTES

# Everything related to calculations uses Float32 (Single Precision). For output related, it uses Float64 (Double precision), examples include Kernel time, Average time, etc.




import JACC
JACC.@init_backend

using Test
using StaticArrays
using InteractiveUtils # To use @code_llvm
using Profile # for profiling
include("BUDE.jl")

const Float32Max = typemax(Float32)
const JACCArray = JACC.array_type()

function run(params::Params, deck::Deck) #_::DeviceWithRepr)
  #println("Using max $(Threads.nthreads()) threads")
  # if params.ppwi != DefaultPPWI
  #   @warn "Threaded implementation only uses wgsize, the PPWI argument is ignored"
  # end

  #nposes::Int = size(deck.poses)[2]
  #poses = size(deck.poses)[2]
  protein = JACCArray{Atom}(deck.protein) #The number of atoms in the protein molecule. The protein is typically the larger molecule that the ligand is being docked to.
  ligand = JACCArray{Atom}(deck.ligand) #The number of atoms in the ligand molecule. The ligand is typically a small molecule that we're trying to dock to the prot
  forcefield = JACCArray{FFParams}(deck.forcefield) # The number of force field parameters used in the simulation. Force fields define the interactions between atoms and are crucial for calculating the energy of each pose.
  poses = JACCArray{Float32,2}(deck.poses) #This indicates the number of different orientations and positions of the ligand molecule being tested
  #etotals = JACCArray{Float32}(undef, poses)


  etotals = JACCArray{Float32}(undef, size(deck.poses)[2]) #stores the total energy for each pose.

  #CUDA.@profile sin.(etotals)

  println("Type of protein: ", typeof(protein))
  #println("protein array location: ", typeof(protein).parameters[3]) #oneAPI.oneL0.DeviceBuffer means its in the GPU.



  # println("Getting LLVM IR...")
  # @code_llvm fasten_main(
  #   Val(convert(Int, params.ppwi)),
  #   protein,
  #   ligand,
  #   forcefield,
  #   poses,
  #   etotals,
  # )


  # warmup
  fasten_main(
    Val(convert(Int, params.ppwi)),
    protein,
    ligand,
    forcefield,
    poses,
    etotals,
  )


  #@code_llvm fasten_main(Val(4), protein, ligand, forcefield, poses, etotals)


  # ORIGINAL ELAPSED FROM THREADED.JL

  # elapsed = @elapsed for _ = 1:params.iterations
  #   fasten_main(
  #     Val(convert(Int, params.wgsize)),
  #     protein,
  #     ligand,
  #     forcefield,
  #     poses,
  #     etotals,
  #   )
  # end





  total_elapsed = 0.0
  for i = 1:params.iterations
    start_time = time()
    begin
      fasten_main(
        Val(convert(Int, params.ppwi)),
        protein,
        ligand,
        forcefield,
        poses,
        etotals,
      )
    end
    end_time = time()
    iteration_elapsed = end_time - start_time
    total_elapsed += iteration_elapsed
    println("Iteration $(i): $(iteration_elapsed) seconds")
  end

  # @profile begin
  #   total_elapsed = 0.0
  #   for i = 1:params.iterations
  #       start_time = time()
  #       fasten_main(
  #           Val(convert(Int, params.ppwi)),
  #           protein,
  #           ligand,
  #           forcefield,
  #           poses,
  #           etotals,
  #       )
  #       end_time = time()
  #       iteration_elapsed = end_time - start_time
  #       total_elapsed += iteration_elapsed
  #       println("Iteration $(i): $(iteration_elapsed) seconds")
  #   end
  # end

  #   # Print profiling results in different formats
  #   println("\nFlat profile (sorted by time):")
  #   Profile.print(format=:flat, sortedby=:count)
    
  #   println("\nTree view of profile:")
  #   Profile.print(maxdepth=30)



  average_time = total_elapsed / params.iterations
  println("Total elapsed time: $(total_elapsed) seconds")
  println("Average time per iteration: $(average_time) seconds")
  

  # Transfer etotals back to CPU (for cuda and amdgpu only)
  etotals_cpu = Array(etotals)

  
  #(etotals, total_elapsed, params.wgsize)
  #(etotals_cpu, elapsed, params.wgsize)
  (etotals_cpu, total_elapsed, params.ppwi)
end


@fastmath function fasten_main(
  ::Val{PPWI},
  protein::AbstractArray{Atom},
  ligand::AbstractArray{Atom},
  forcefield::AbstractArray{FFParams},
  poses::AbstractArray{Float32,2},
  etotals::AbstractArray{Float32},
) where {PPWI}

  nposes::Int = size(poses)[2]
  nligand::Int = length(ligand)
  nprotein::Int = length(protein)
  total_work_items::Int = ceil(Int, nposes / PPWI)
  #total_work_items = 1000
  #print("total_work_items:", total_work_items)
  

  function kernel(index)
    # Get index of first TD (Thread Data)
    ix = (index - 1) * PPWI + 1 #calculates the starting pose index for each work item
    ix = ix <= nposes ? ix : nposes - PPWI #make sure we dont go out of bounds (not go past)

    etot = MArray{Tuple{PPWI},Float32}(undef)
    transform = MArray{Tuple{PPWI,3},Vec4f32}(undef)

    @simd for i = 1:PPWI
      pose_index = ix + i - 1
      @inbounds sx::Float32 = sin(poses[1, pose_index])
      @inbounds cx::Float32 = cos(poses[1, pose_index])
      @inbounds sy::Float32 = sin(poses[2, pose_index])
      @inbounds cy::Float32 = cos(poses[2, pose_index])
      @inbounds sz::Float32 = sin(poses[3, pose_index])
      @inbounds cz::Float32 = cos(poses[3, pose_index])
      @inbounds transform[i, 1] = #
        Vec4f32(cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz, poses[4, pose_index])
      @inbounds transform[i, 2] = #
        Vec4f32(cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz, poses[5, pose_index])
      @inbounds transform[i, 3] = #
        Vec4f32(-sy, sx * cy, cx * cy, poses[6, pose_index])
      @inbounds etot[i] = 0
    end

    #for il = 1:nligand
    @inbounds for l_atom::Atom in ligand
      #l_atom = ligand[il]
      l_params = forcefield[l_atom.type+1]
      lhphb_ltz = l_params.hphb < Zero
      lhphb_gtz = l_params.hphb > Zero

      lpos = MArray{Tuple{PPWI},Vec3f32}(undef)

    @simd for i = 1:PPWI
      # Transform ligand atom
      @inbounds lpos[i] = Vec3f32(
        transform[i, 1].w +
        l_atom.x * transform[i, 1].x +
        l_atom.y * transform[i, 1].y +
        l_atom.z * transform[i, 1].z,
        transform[i, 2].w +
        l_atom.x * transform[i, 2].x +
        l_atom.y * transform[i, 2].y +
        l_atom.z * transform[i, 2].z,
        transform[i, 3].w +
        l_atom.x * transform[i, 3].x +
        l_atom.y * transform[i, 3].y +
        l_atom.z * transform[i, 3].z,
      )
    end

      #for ip = 1:nprotein
      @inbounds for p_atom::Atom in protein
        #p_atom = protein[ip]
        #p_params = forcefield[p_atom.type+1]
        @inbounds p_params::FFParams = forcefield[p_atom.type+1]


        radij::Float32 = p_params.radius + l_params.radius
        r_radij::Float32 = One / radij
  
        elcdst::Float32 = (p_params.hbtype == HbtypeF && l_params.hbtype == HbtypeF) ? Four : Two
        elcdst1::Float32 = (p_params.hbtype == HbtypeF && l_params.hbtype == HbtypeF) ? Quarter : Half
        type_E::Bool = ((p_params.hbtype == HbtypeE || l_params.hbtype == HbtypeE))
  
        phphb_ltz::Bool = p_params.hphb < Zero
        phphb_gtz::Bool = p_params.hphb > Zero
        phphb_nz::Bool = p_params.hphb != Zero
        p_hphb::Float32 = p_params.hphb * (phphb_ltz && lhphb_gtz ? -One : One)
        l_hphb::Float32 = l_params.hphb * (phphb_gtz && lhphb_ltz ? -One : One)
        distdslv::Float32 =
          (phphb_ltz ? (lhphb_ltz ? Npnpdist : Nppdist) : (lhphb_ltz ? Nppdist : -Float32Max))
        r_distdslv::Float32 = One / distdslv
  
        chrg_init::Float32 = l_params.elsc * p_params.elsc
        dslv_init::Float32 = p_hphb + l_hphb

      @simd for i = 1:PPWI
        @inbounds x::Float32 = lpos[i].x - p_atom.x
        @inbounds y::Float32 = lpos[i].y - p_atom.y
        @inbounds z::Float32 = lpos[i].z - p_atom.z

        distij::Float32 = sqrt(x ^ 2 + y ^ 2 + z ^ 2)

        # Calculate the sum of the sphere radii
        distbb::Float32 = distij - radij
        zone1::Bool = (distbb < Zero)

        # Calculate steric energy
        @inbounds etot[i] += (One - (distij * r_radij)) * (zone1 ? Two * Hardness : Zero)

        # Calculate formal and dipole charge interactions
        chrg_e::Float32 =
          chrg_init * ((zone1 ? One : (One - distbb * elcdst1)) * (distbb < elcdst ? One : Zero))
        neg_chrg_e::Float32 = -abs(chrg_e)
        chrg_e = type_E ? neg_chrg_e : chrg_e
        @inbounds etot[i] += chrg_e * Cnstnt

        # Calculate the two cases for Nonpolar-Polar repulsive interactions
        coeff::Float32 = (One - (distbb * r_distdslv))
        dslv_e::Float32 = dslv_init * ((distbb < distdslv && phphb_nz) ? One : Zero)
        dslv_e *= (zone1 ? One : coeff)
        @inbounds etot[i] += dslv_e
      end
    end
  end

    @simd for i = 1:PPWI
      pose_index = ix + i - 1
      if pose_index <= nposes
        @inbounds etotals[pose_index] = etot[i] * Half
      end
    end
  end #end for kernel function

  JACC.parallel_for(total_work_items, kernel) # total_work_items determines how the pose indices are calculated
  # example: if total_work_items is 25, then kernel(index) is called 25 times when doing JACC.parallel_for.
end

main()


