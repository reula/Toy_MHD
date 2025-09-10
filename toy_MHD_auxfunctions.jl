function Div(B,Dx,Dy,J) 
    div = zeros(J[1],J[2])
    @threads for j in 1:J[2]
        div[:,j] .+= Dx*B[1,:,j]
    end
    @threads for i in 1:J[1]
        div[i,:] .+= Dy*B[2,i,:]
    end
    return div
end 

function F_par!(u, t, par)
    J, Dx, Δx, Dy, Δy, σx, σy, V, κ, c2, τ, du = par
    du .=0.0
    u_a = reshape(u,(D+1,J...))
    Du_a = reshape(du,(D+1,J...))
    div = zeros(J...)

    #first the divergence cleaning term
    @threads for j in 1:J[2]
        mul!(view(div,:,j), Dx, view(u_a,1,:,j))
    end
    @threads for i in 1:J[1]
        mul!(view(div,i,:), Dy, view(u_a,2,i,:), one(eltype(u_a)), one(eltype(u_a)))
    end
    # the y derivative term 
    @threads for i in 1:J[1]
        mul!(view(Du_a,1,i,:), Dy, view(u_a,1,i,:).*V[2,i,:] - view(u_a,2,i,:).*V[1,i,:])
        mul!(view(Du_a,2,i,:), Dy, view(div,i,:), κ, zero(eltype(u_a)))
    end
    @threads for j in 1:J[2]
        mul!(view(Du_a,2,:,j), Dx , view(u_a,2,:,j).*V[1,:,j] - view(u_a,1,:,j).*V[2,:,j], one(eltype(u_a)), one(eltype(u_a)))
        mul!(view(Du_a,1,:,j), Dx , view(div,:,j), κ, one(eltype(u_a)))
    end
    #=
    @threads for i in 1:J[1]
        mul!(view(Du_a,1,i,:), Δy, view(u_a,1,i,:), σy, one(eltype(u_a)))
        mul!(view(Du_a,2,i,:), Δy, view(u_a,2,i,:), σy, one(eltype(u_a)))
    end
    @threads for j in 1:J[2]
        mul!(view(Du_a,1,:,j), Δx, view(u_a,1,:,j), σx, one(eltype(u_a)))
        mul!(view(Du_a,2,:,j), Δx, view(u_a,2,:,j), σx, one(eltype(u_a)))
    end
    =#
    return du[:]
end

function F_hyp!(u, t, par)
    J, Dx, Δx, Dy, Δy, σx, σy, V, κ, τ, c2, du = par
    du .=0.0
    u_a = reshape(u,(D+1,J...))
    Du_a = reshape(du,(D+1,J...))
    #div = zeros(J...)
    
    # the y derivative term 
    @threads for i in 1:J[1]
        mul!(view(Du_a,1,i,:), Dy, view(u_a,1,i,:).*V[2,i,:] - view(u_a,2,i,:).*V[1,i,:])
        mul!(view(Du_a,2,i,:), Dy, view(u_a,D+1,i,:), one(eltype(u_a)), zero(eltype(u_a)))
        mul!(view(Du_a,D+1,i,:), Dy, view(u_a,2,i,:), c2, zero(eltype(u_a)))
        Du_a[D+1,i,:] .+= - τ*u_a[D+1,i,:]
    end
    # the x derivative terms
    @threads for j in 1:J[2]
        mul!(view(Du_a,2,:,j), Dx , view(u_a,2,:,j).*V[1,:,j] - view(u_a,1,:,j).*V[2,:,j], one(eltype(u_a)), one(eltype(u_a)))
        mul!(view(Du_a,D+1,:,j), Dx, view(u_a,1,:,j), c2, one(eltype(u_a)))
        mul!(view(Du_a,1,:,j), Dx , view(u_a,D+1,:,j), one(eltype(u_a)), one(eltype(u_a)))
    end
    #=
    @threads for i in 1:J[1]
        mul!(view(Du_a,1,i,:), Δy, view(u_a,1,i,:), σy, one(eltype(u_a)))
        mul!(view(Du_a,2,i,:), Δy, view(u_a,2,i,:), σy, one(eltype(u_a)))
    end
    @threads for j in 1:J[2]
        mul!(view(Du_a,1,:,j), Δx, view(u_a,1,:,j), σx, one(eltype(u_a)))
        mul!(view(Du_a,2,:,j), Δx, view(u_a,2,:,j), σx, one(eltype(u_a)))
    end
    =#
    return du[:]
end

function get_Energy(u,J,Box)
    return norm(u)^2*volume(Box)/prod(J)
end

