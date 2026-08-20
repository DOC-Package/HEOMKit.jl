@testset "BLAS helpers" begin
    cfg = blas_config()
    @test occursin("LBTConfig", sprint(show, cfg))

    old_threads = blas_num_threads()
    @test old_threads >= 1

    @test set_blas_threads!(1) == 1
    @test blas_num_threads() == 1
    @test_throws ArgumentError set_blas_threads!(0)

    set_blas_threads!(old_threads)
    @test blas_num_threads() == old_threads
end
