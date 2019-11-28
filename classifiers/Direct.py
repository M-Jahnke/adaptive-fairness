from dataclasses import dataclass
import numpy as np

@dataclass
class Options:
    maxits: int = 20
    maxevals: int = 10
    maxdeep: int = 100
    testflag: int = 0
    globalmin: int = 0
    ep: float = 1e-4
    tol: float = 0.01
    showits: int = 1
    impcons: int = 0
    pert: float = 1e-6

def Direct(Problem, bounds, varargin, opts=Options()):

    lengths = []
    c = []
    fc = []
    con = []
    szes = []
    feas_flags=[]
    om_lower     = bounds[:,0]
    om_upper     = bounds[:,1]
    fcncounter   = 0
    perror       = 0
    itctr        = 1
    done         = 0
    g_nargout    = nargout
    n            = np.size(bounds, 0) #wrong

    # Determine option values
    # if nargin<3, opts=[]; end
    # if (nargin>=3) & (length(opts)==0), opts=[]; end
    if len(opts.__annotations__) == 0:
        opts = Options()

    theglobalmin = opts.globalmin
    tflag        = opts.testflag

    #-- New 06/08/2004 Pre-allocate memory for storage vectors
    if tflag == 0:
        lengths    = np.zeros(n, opts.maxevals + np.floor(.10 * opts.maxevals))
        c          = lengths
        fc         = np.zeros(1, opts.maxevals + np.floor(.10*opts.maxevals))
        szes       = fc
        con        = fc
        feas_flags = fc


    #%-- Call DIRini ---------------------------------------------------%
    thirds , lengths, c , fc, con, feas_flags, minval,xatmin,perror,history,szes,fcncounter,calltype = DIRini(Problem,n,bounds[:,0],bounds[:,1],lengths,c,fc,con, feas_flags, szes,theglobalmin,opts.maxdeep,tflag,g_nargout, opts.impcons, varargin)

    ret_minval = minval
    ret_xatmin = xatmin
    #-- MAIN LOOP -----------------------------------------------------
    minval = fc[0] + con[1]
    while perror > opts.tol:
       #-- Create list S of potentially optimal hyper-rectangles
       S = find_po(fc[0:fcncounter]+con[0:fcncounter], lengths[:,1:fcncounter],minval,opts.ep,szes[0:fcncounter])

       #-- Loop through the potentially optimal hrectangles -----------
       #-- and divide -------------------------------------------------
       #for i = 1:size(S,2)
       for i in range(np.size[S, 1]):
          lengths,fc,c,con,feas_flags,szes,fcncounter,success = DIRdivide(bounds[:,0],bounds[:,1],Problem,S[0,i],thirds,lengths, fc,c,con,feas_flags,fcncounter,szes,impcons,calltype,varargin)

       #-- update minval, xatmin --------------------------------------
       [minval,fminindex] =  min(fc[0:fcncounter]+con[0:fcncounter])
       penminval = minval + con[fminindex]
       xatmin = (om_upper - om_lower) * c[:,fminindex] + om_lower
       if (con[fminindex] > 0) or (feas_flags[fminindex] != 0):
           #--- new minval is infeasible, don't do anything
            pass
       else:
           #--- update return values
           ret_minval = minval
           ret_xatmin = xatmin

       #--see if we are done ------------------------------------------
       if tflag == 1:
          #-- Calculate error if globalmin known
          if theglobalmin != 0:
              perror = 100*(minval - theglobalmin)/abs(theglobalmin)
          else:
              perror = 100*minval
       else:
          #-- Have we exceeded the maxits?
          if itctr >= opts.maxits:
             print('Exceeded max iterations. Increase maxits')
             done = 1
          #-- Have we exceeded the maxevals?
          if fcncounter > opts.maxevals:
             print('Exceeded max fcn evals. Increase maxevals')
             done = 1
          if done == 1:
             perror = -1
       if max(max(lengths)) >= opts.maxdeep:
          #-- We've exceeded the max depth
          print('Exceeded Max depth. Increse maxdeep')
          perror = -1
       if g_nargout == 3:
          #-- Store History
          maxhist = np.size(history,0)
          #history = history[history + [0] for x in L]
          history[maxhist+1,0] = itctr
          history[maxhist+1,1] = fcncounter
          history[maxhist+1,2] = minval

      #-- New, 06/09/2004
      #-- Call replaceinf if impcons flag is set to 1
      if opts.impcons == 1:
          fc = replaceinf(lengths(:,1:fcncounter),c(:,1:fcncounter),...
              fc(1:fcncounter),con(1:fcncounter),...
              feas_flags(1:fcncounter),pert);
      end

      #-- show iteration stats
      if showits == 1
        if  (con(fminindex) > 0) | (feas_flags(fminindex) == 1)
            fprintf('Iter: %4i   f_min: %15.10f*    fn evals: %8i\n',...
             itctr,minval,fcncounter);
        else
            fprintf('Iter: %4i   f_min: %15.10f    fn evals: %8i\n',...
             itctr,minval,fcncounter);
        end
      end
      itctr  = itctr + 1;
    end

    #-- Return values
    if g_nargout == 2
        #-- return x*
        final_xatmin = ret_xatmin;
    elseif g_nargout == 3
        #-- return x*
        final_xatmin = ret_xatmin;

        #-- chop off 1st row of history
        history(1:size(history,1)-1,:) = history(2:size(history,1),:);
        history = history(1:size(history,1)-1,:);
    end
    return [ret_minval,  final_xatmin,history]


function [l_thirds,l_lengths,l_c,l_fc,l_con, l_feas_flags, minval,xatmin,perror,...
        history,szes,fcncounter,calltype] = DIRini(Problem,n,a,b,...
        p_lengths,p_c,p_fc,p_con, p_feas_flags, p_szes,theglobalmin,...
        maxdeep,tflag,g_nargout,impcons,varargin)

l_lengths    = p_lengths;
l_c          = p_c;
l_fc         = p_fc;
l_con        = p_con;
l_feas_flags = p_feas_flags;
szes         = p_szes;


#-- start by calculating the thirds array
#-- here we precalculate (1/3)^i which we will use frequently
l_thirds(1) = 1/3;
for i = 2:maxdeep
   l_thirds(i) = (1/3)*l_thirds(i-1);
end

#-- length array will store # of slices in each dimension for
#-- each rectangle. dimension will be rows; each rectangle
#-- will be a column

#-- first rectangle is the whole unit hyperrectangle
l_lengths(:,1) = zeros(n,1);

#01/21/04 HACK
#-- store size of hyperrectangle in vector szes
szes(1,1) = 1;

#-- first element of c is the center of the unit hyperrectangle
l_c(:,1) = ones(n,1)/2;

#-- Determine if there are constraints
calltype = DetermineFcnType(Problem,impcons);

#-- first element of f is going to be the function evaluated
#-- at the center of the unit hyper-rectangle.
#om_point   = abs(b - a).*l_c(:,1)+ a;
#l_fc(1)    = feval(f,om_point,varargin{:});
[l_fc(1),l_con(1), l_feas_flags(1)] = ...
    CallObjFcn(Problem,l_c(:,1),a,b,impcons,calltype,varargin{:});
fcncounter = 1;


#-- initialize minval and xatmin to be center of hyper-rectangle
xatmin = l_c(:,1);
minval   = l_fc(1);
if tflag == 1
    if theglobalmin ~= 0
        perror = 100*(minval - theglobalmin)/abs(theglobalmin);
    else
        perror = 100*minval;
    end
else
   perror = 2;
end

#-- initialize history
#if g_nargout == 3
    history(1,1) = 0;
    history(1,2) = 0;
    history(1,3) = 0;
#end


function rects = find_po(fc,lengths,minval,ep,szes)

%-- 1. Find all rects on hub
diff_szes = sum(lengths,1);
tmp_max = max(diff_szes);
j=1;
sum_lengths = sum(lengths,1);
for i =1:tmp_max+1
    tmp_idx = find(sum_lengths==i-1);
    [tmp_n, hullidx] = min(fc(tmp_idx));
    if length(hullidx) > 0
        hull(j) = tmp_idx(hullidx);
        j=j+1;
        %-- 1.5 Check for ties
        ties = find(abs(fc(tmp_idx)-tmp_n) <= 1e-13);
        if length(ties) > 1
            mod_ties = find(tmp_idx(ties) ~= hull(j-1));
            hull = [hull tmp_idx(ties(mod_ties))];
            j = length(hull)+1;
        end
    end
end
%-- 2. Compute lb and ub for rects on hub
lbound = calc_lbound(lengths,fc,hull,szes);
ubound = calc_ubound(lengths,fc,hull,szes);

%-- 3. Find indeces of hull who satisfy
%--    1st condition
maybe_po = find(lbound-ubound <= 0);

%-- 4. Find indeces of hull who satisfy
%--    2nd condition
t_len  = length(hull(maybe_po));
if minval ~= 0
    po = find((minval-fc(hull(maybe_po)))./abs(minval) +...
        szes(hull(maybe_po)).*ubound(maybe_po)./abs(minval) >= ep);
else
    po = find(fc(hull(maybe_po)) -...
        szes(hull(maybe_po)).*ubound(maybe_po) <= 0);
end
final_pos      = hull(maybe_po(po));

rects = [final_pos;szes(final_pos)];
return


function ub = calc_ubound(lengths,fc,hull,szes)

hull_length  = length(hull);
hull_lengths = lengths(:,hull);
for i =1:hull_length
    tmp_rects = find(sum(hull_lengths,1)<sum(lengths(:,hull(i))));
    if length(tmp_rects) > 0
        tmp_f     = fc(hull(tmp_rects));
        tmp_szes  = szes(hull(tmp_rects));
        tmp_ubs   = (tmp_f-fc(hull(i)))./(tmp_szes-szes(hull(i)));
        ub(i)        = min(tmp_ubs);
    else
        ub(i)=1.976e14;
    end
end
return


function lb = calc_lbound(lengths,fc,hull,szes)

hull_length  = length(hull);
hull_lengths = lengths(:,hull);
for i = 1:hull_length
    tmp_rects = find(sum(hull_lengths,1)>sum(lengths(:,hull(i))));
    if length(tmp_rects) > 0
        tmp_f     = fc(hull(tmp_rects));
        tmp_szes  = szes(hull(tmp_rects));
        tmp_lbs   = (fc(hull(i))-tmp_f)./(szes(hull(i))-tmp_szes);
        lb(i)     = max(tmp_lbs);
    else
        lb(i)     = -1.976e14;
    end
end
return


function [lengths,fc,c,con,feas_flags,szes,fcncounter,pass] = ...
    DIRdivide(a,b,Problem,index,thirds,p_lengths,p_fc,p_c,p_con,...
    p_feas_flags,p_fcncounter,p_szes,impcons,calltype,varargin)

lengths    = p_lengths;
fc         = p_fc;
c          = p_c;
szes       = p_szes;
fcncounter = p_fcncounter;
con        = p_con;
feas_flags = p_feas_flags;

%-- 1. Determine which sides are the largest
li     = lengths(:,index);
biggy  = min(li);
ls     = find(li==biggy);
lssize = length(ls);
j = 0;

%-- 2. Evaluate function in directions of biggest size
%--    to determine which direction to make divisions
oldc       = c(:,index);
delta      = thirds(biggy+1);
newc_left  = oldc(:,ones(1,lssize));
newc_right = oldc(:,ones(1,lssize));
f_left     = zeros(1,lssize);
f_right    = zeros(1,lssize);
for i = 1:lssize
    lsi               = ls(i);
    newc_left(lsi,i)  = newc_left(lsi,i) - delta;
    newc_right(lsi,i) = newc_right(lsi,i) + delta;
    [f_left(i), con_left(i), fflag_left(i)]    = CallObjFcn(Problem,newc_left(:,i),a,b,impcons,calltype,varargin{:});
    [f_right(i), con_right(i), fflag_right(i)] = CallObjFcn(Problem,newc_right(:,i),a,b,impcons,calltype,varargin{:});
    fcncounter = fcncounter + 2;
end
w = [min(f_left, f_right)' ls];

%-- 3. Sort w for division order
[V,order] = sort(w,1);

%-- 4. Make divisions in order specified by order
for i = 1:size(order,1)

   newleftindex  = p_fcncounter+2*(i-1)+1;
   newrightindex = p_fcncounter+2*(i-1)+2;
   %-- 4.1 create new rectangles identical to the old one
   oldrect = lengths(:,index);
   lengths(:,newleftindex)   = oldrect;
   lengths(:,newrightindex)  = oldrect;

   %-- old, and new rectangles have been sliced in order(i) direction
   lengths(ls(order(i,1)),newleftindex)  = lengths(ls(order(i,1)),index) + 1;
   lengths(ls(order(i,1)),newrightindex) = lengths(ls(order(i,1)),index) + 1;
   lengths(ls(order(i,1)),index)         = lengths(ls(order(i,1)),index) + 1;

   %-- add new columns to c
   c(:,newleftindex)  = newc_left(:,order(i));
   c(:,newrightindex) = newc_right(:,order(i));

   %-- add new values to fc
   fc(newleftindex)  = f_left(order(i));
   fc(newrightindex) = f_right(order(i));

   %-- add new values to con
   con(newleftindex)  = con_left(order(i));
   con(newrightindex) = con_right(order(i));

   %-- add new flag values to feas_flags
   feas_flags(newleftindex)  = fflag_left(order(i));
   feas_flags(newrightindex) = fflag_right(order(i));

   %-- 01/21/04 Dan Hack
   %-- store sizes of each rectangle
   szes(1,newleftindex)  = 1/2*norm((1/3*ones(size(lengths,1),1)).^(lengths(:,newleftindex)));
   szes(1,newrightindex) = 1/2*norm((1/3*ones(size(lengths,1),1)).^(lengths(:,newrightindex)));
end
szes(index) = 1/2*norm((1/3*ones(size(lengths,1),1)).^(lengths(:,index)));
pass = 1;

return


function ret_value = CallConstraints(Problem,x,a,b,varargin)

%-- Scale variable back to original space
point = abs(b - a).*x+ a;

ret_value = 0;
if isfield(Problem,'constraint')
    if ~isempty(Problem.constraint)
        for i = 1:Problem.numconstraints
            if length(Problem.constraint(i).func) == length(Problem.f)
                if double(Problem.constraint(i).func) == double(Problem.f)
                    %-- Dont call constraint; value was returned in obj fcn
                    con_value = 0;
                else
                    con_value = feval(Problem.constraint(i).func,point,varargin{:});
                end
            else
                con_value = feval(Problem.constraint(i).func,point,varargin{:});
            end
            if con_value > 0
                %-- Infeasible, punish with associated pen. param
                ret_value = ret_value + con_value*Problem.constraint(i).penalty;
            end
        end
    end
end
return


function [fcn_value, con_value, feas_flag] = ...
    CallObjFcn(Problem,x,a,b,impcon,calltype,varargin)

con_value = 0;
feas_flag = 0;

%-- Scale variable back to original space
point = abs(b - a).*x+ a;

if calltype == 1
    %-- No constraints at all
    fcn_value = feval(Problem.f,point,varargin{:});
elseif calltype == 2
    %-- f returns all constraints
    [fcn_value, cons] = feval(Problem.f,point,varargin{:});
    for i = 1:length(cons)
        if cons > 0
            con_value = con_value + Problem.constraint(i).penalty*cons(i);
        end
    end
elseif calltype == 3
    %-- f returns no constraint values
    fcn_value = feval(Problem.f,point,varargin{:});
    con_value = CallConstraints(Problem,x,a,b,varargin{:});
elseif calltype == 4
    %-- f returns feas flag
    [fcn_value,feas_flag] = feval(Problem.f,point,varargin{:});
elseif calltype == 5
    %-- f returns feas flags, and there are constraints
    [fcn_value,feas_flag] = feval(Problem.f,point,varargin{:});
    con_value = CallConstraints(Problem,x,a,b,varargin{:});
end
if feas_flag == 1
    fcn_value = 10^9;
    con_value = 0;
end


def replaceinf(lengths,c,fc,con,flags,pert):

    #-- Initialize fcn_values to original values
    fcn_values = fc

    #-- Find the infeasible points
    # infeas_points = find(flags == 1) # (MATLAB) find(a) <==> a.ravel().nonzero() (Python)
    mflags = np.asarray([True if flags[i] == 1 else False for i in range(0, np.size(flags, 0))])
    infeas_points = mflags.ravel().nonzero()

    #-- Find the feasible points
    # feas_points = find(flags == 0)
    mflags = np.asarray([True if flags[i] == 0 else False for i in range(0, np.size(flags, 0))])
    feas_points = mflags.ravel().nonzero()

    #-- Calculate the max. value found so far
    #if not isempty(feas_points):
    if feas_points.size == 0:
        maxfc = max(fc[feas_points] + con[feas_points])
    else:
        maxfc = max(fc + con)

    #for i = 1:length(infeas_points)
    for i in range(0, np.size(infeas_points, 0)):
        #if isempty(feas_points):
        if feas_points.size == 0:
            #-- no feasible points found yet
            found_points = []
            found_pointsf = []
            index = infeas_points[i]
        else:
            index = infeas_points[i]

            #-- Initialize found points to be entire set
            found_points  = c[:, feas_points - 1]
            found_pointsf = fc[feas_points] + con[feas_points]

            #-- Loop through each dimension, and find points who are close enough
            #for j = 1:size(lengths,1)
            for j in range(0, np.size(lengths, 0)):
                # neighbors = find(abs(found_points(j,:) - c(j,index)) <= 3^(-lengths(j,index)))
                mNeighbors = np.asarray([True if abs(found_points[j,k] - c[j, index]) <= 3 ^ (-lengths[j, index]) else
                                         False for k in range(0, np.size(found_points, 0))])
                neighbors = mNeighbors.ravel().nonzero()
                #if not isempty(neighbors):
                if not neighbors.size == 0:
                    found_points  = found_points[:, neighbors]
                    found_pointsf = found_pointsf[neighbors]
                else:
                    found_points = [];found_pointsf = []
                    break

        #-- Assign Carter value to the point
        #if not isempty(found_pointsf):
        if not found_pointsf.size == 0:
            #-- assign to index the min. value found + a little bit more
            fstar = min(found_pointsf)
            if fstar != 0:
                fcn_values[index] = fstar + pert*abs(fstar)
            else:
                fcn_values[index] = fstar + pert*1
        else:
            fcn_values[index] = maxfc+1
            maxfc = maxfc+1

    return fcn_values

def DetermineFcnType(Problem, impcons):
    retval = 0

    #if (not isfield(Problem, 'constraint')) and (not impcons):
    if (not ('constrains' in Problem.__annotations__)) and (not impcons):
        #-- No constraints at all
        retval = 1

    '''
    #if isfield(Problem,'constraint'):
    if 'constraint' in Problem.__annotations__: # wir haben nie constraints <----- !!! also ist diese if irrelevant
        #-- There are explicit constraints. Next determine where
        #-- they are called
        #if not isempty(Problem.constraint):
        if not Problem.constraint == None:
            if length(Problem.constraint(1).func) == length(Problem.f): # length meint hier die Anzahl der an das Problem zu übergebenden Parameter, z.B: f(x, y) hat Länge 2
                #-- Constraint values may be returned from objective
                #-- function. Investigate further
                if double(Problem.constraint(1).func) == double(Problem.f):
                    #-- f returns constraint values
                    retval = 2
                else:
                    #-- f does not return constraint values
                    retval = 3

            else:
                #-- f does not return constraint values
                retval = 3

        else:
            if impcons:
                retval = 0
            else:
                retval = 1
    '''

    if (impcons):
        if not retval:
            #-- only implicit constraints
            retval = 4
        else:
            #-- both types of constraints
            retval = 5

    return retval