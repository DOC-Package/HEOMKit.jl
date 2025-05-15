
function kropro_id_left(A::AbstractMatrix{ComplexF64})
    n = length(axes(A,1))
    return kron(A, I(n))
    
kropro_id_left(A::AbstractMatrix)  = kron(A, I(size(A,1)))
kropro_id_right(A::AbstractMatrix) = kron(I(size(A,1)), A)

matx(A::AbstractMatrix) = kropro_id_left(A) - kropro_id_right(adjoint(A))  
mato(A::AbstractMatrix) = kropro_id_left(A) + kropro_id_right(adjoint(A))  
matl(A::AbstractMatrix) = kropro_id_left(A)
