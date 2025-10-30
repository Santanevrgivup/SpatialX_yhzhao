#####################################################################
## Package: SPATIALX, 
## Package includes: SpatialVG, for single-slice
## Version: 0.1.0
## Date: 2020-2-14
## Modified: 2020-2-22 04:50:39;2020-3-14 19:55:43;2022-6-4 09:26:03;2022-9-17 09:37:51;
## 2024-8-13 21:45:39
## Title : Spatially-aware expression analysis of spatially resolved transcriptomics
## Determining the spatially variable genes using Poisson mixed models and 
## spatially variable genes using logistic mixed models
## Authors: S.Q. Sun
## Contacts: sqsunsph@xjtu.edu.cn
##          Xi'an Jiaotong University, 
##          Center for Single-Cell Omics and Health, Department of Biostatistics
######################################################################


#' Spatially Variable Gene Identification
#'
#' These functions perform spatially variable gene (SVG) identification using
#' various statistical models and kernel methods. They support multiple data
#' types including raw counts, normalized data, and binary expression patterns,
#' and can handle different spatial correlation structures through flexible
#' kernel specifications.
#'
#' @param object Input data object. Can be a matrix, SpatialX object, Assay object,
#'              Seurat object, or SpatialExperiment object containing gene expression
#'              data and spatial coordinates.
#' @param svg.method Method for SVG identification. Options include:
#'   \itemize{
#'     \item "spark": Poisson model with mixture kernels (Nature Methods, 2020)
#'     \item "sparkx": Non-parametric model with mixture kernels
#'     \item "sparkg": Gaussian model with mixture kernels
#'     \item "csvg": Cell-type specific spatially variable genes
#'     \item "dsvg": Domain-specific spatially variable genes
#'     \item "isvg": Integrative spatially variable genes
#'   }
#' @param covariates Covariates to adjust for (e.g., batch effects, confounding factors).
#'                  Currently limited support due to computational intensity.
#' @param location Spatial coordinates matrix or data.frame with x and y coordinates
#'                for each cell/spot.
#' @param cell.prop Cell type proportion matrix for cell-type specific analysis.
#' @param cell.type Specific cell type(s) to test. If NULL, tests all available cell types.
#' @param lib.size Library size (total counts) for each cell/spot. Used for count-based
#'                models to account for sequencing depth variation.
#' @param kernel.mat Precomputed kernel matrices as a list. If provided, skips
#'                  kernel computation and uses these matrices directly.
#' @param kernel.type Type of kernel function(s) to use. Can be "gaussian", "cosine",
#'                   "linear", "matern", "spline", or a mixture of these.
#' @param kernel.param Parameters for kernel functions. For Gaussian kernel, this is
#'                    the bandwidth; for polynomial, the degree, etc.
#' @param check.positive Whether to check and ensure kernel matrices are positive
#'                      definite. Recommended for numerical stability.
#' @param fit.model Statistical model to use for testing:
#'   \itemize{
#'     \item "poisson": Count-based model for raw counts
#'     \item "binomial": Binary model for binarized expression
#'     \item "gaussian": Gaussian model for normalized data
#'     \item "nonparametric": Non-parametric model (SPARK-X)
#'   }
#' @param fit.maxiter Maximum number of iterations for model fitting.
#' @param fit.tol Tolerance threshold for model convergence.
#' @param weights Weights for combining p-values from multiple kernels. If NULL,
#'               uses equal weights.
#' @param num.core Number of CPU cores to use for parallel computation.
#' @param verbose Whether to print progress messages and debugging information.
#' @param assay Name of assay to use (for SpatialX and Seurat objects).
#' @param slot.use Data slot to use for analysis. For count models use "counts",
#'                for normalized models use "data" or "scale.data".
#' @param features Specific genes/features to test. If NULL, tests all available features.
#' @param top.features Number of top SVGs to return and store.
#' @param svg.param List of additional parameters for SVG analysis.
#' @param kernel.param List of additional parameters for kernel computation.
#' @param ... Additional arguments passed to methods.
#'
#' @return
#' For \code{SpatialVG.default}: Returns a data.frame with combined p-values,
#' test statistics, and adjusted p-values for each gene, ordered by significance.
#'
#' For \code{SpatialVGC.default}: Returns a list of data.frames, one for each
#' cell type, containing SVG analysis results.
#'
#' For object methods (\code{SpatialVG.SpatialX}, \code{SpatialVG.Assay}, etc.):
#' Returns the input object with SVG results stored in the appropriate slots.
#'
#' @details
#' These functions implement sophisticated statistical methods for identifying
#' genes that exhibit spatially correlated expression patterns. Key features:
#'
#' \itemize{
#'   \item \strong{Multiple Statistical Models}: Supports count-based (Poisson),
#'         binary (binomial), normalized (Gaussian), and non-parametric models
#'         to handle different data types and distributions.
#'   \item \strong{Flexible Kernel Methods}: Uses multiple spatial kernels to
#'         capture different spatial correlation patterns at various scales.
#'   \item \strong{Cell-Type Specific Analysis}: Can identify SVGs specific to
#'         particular cell types using cell type proportion information.
#'   \item \strong{Efficient Computation}: Implements parallel processing and
#'         optimized algorithms for handling large spatial datasets.
#'   \item \strong{Comprehensive Multiple Testing}: Combines p-values from
#'         multiple kernels using the ACAT method to provide robust results.
#' }
#'
#' The method works by:
#' \enumerate{
#'   \item Fitting a null model that accounts for technical variation and
#'         covariates (if provided)
#'   \item Computing spatial kernels that capture different spatial scales
#'         and patterns
#'   \item Performing variance component tests for each gene and kernel
#'   \item Combining evidence across kernels using weighted p-value combination
#'   \item Adjusting for multiple testing across genes
#' }
#'
#' @section Model Selection:
#' Choose the appropriate model based on your data:
#' \itemize{
#'   \item \strong{Raw UMI counts}: Use "poisson" model
#'   \item \strong{Binarized expression}: Use "binomial" model
#'   \item \strong{Normalized/log-transformed data}: Use "gaussian" model
#'   \item \strong{Large datasets with complex spatial patterns}: Use "nonparametric"
#' }
#'
#' @section Kernel Selection:
#' Different kernels capture different spatial patterns:
#' \itemize{
#'   \item \strong{Gaussian}: Captures smooth, continuous spatial patterns
#'   \item \strong{Cosine}: Captures periodic spatial patterns
#'   \item \strong{Matern}: Flexible family capturing various smoothness levels
#'   \item \strong{Spline}: Captures complex non-linear spatial patterns
#'   \item \strong{Linear}: Captures linear spatial trends
#' }
#'
#' Using multiple kernels (mixture) is recommended to capture diverse spatial
#' patterns at different scales.
#'
#' @examples
#' \dontrun{
#' # Using a count matrix and spatial coordinates
#' counts <- matrix(rnbinom(10000, mu = 5, size = 2), nrow = 1000, ncol = 10)
#' locations <- matrix(runif(20), ncol = 2)
#'
#' # Basic SVG analysis with default parameters
#' svg_results <- SpatialVG.default(
#'   object = counts,
#'   location = locations,
#'   fit.model = "poisson"
#' )
#'
#' # Cell-type specific SVG analysis
#' cell_props <- matrix(runif(30), ncol = 3)
#' colnames(cell_props) <- c("Neuron", "Astrocyte", "Microglia")
#'
#' csvg_results <- SpatialVGC.default(
#'   object = counts,
#'   cell.prop = cell_props,
#'   location = locations,
#'   cell.type = c("Neuron", "Astrocyte")
#' )
#'
#' # Using with SpatialX object
#' spatialx_obj <- CreateSpatialXObject(
#'   counts = counts,
#'   location = locations
#' )
#'
#' spatialx_with_svg <- SpatialVG(
#'   object = spatialx_obj,
#'   svg.method = "spark",
#'   top.features = 1000
#' )
#'
#' # Access results
#' svg_genes <- GetAssayData(spatialx_with_svg, slot = "svg")
#' top_svg <- GetAssayData(spatialx_with_svg, slot = "sv.genes")
#' }
#'
#' @references
#' Sun, S., Zhu, J., & Zhou, X. (2020). Statistical analysis of spatial expression
#' patterns for spatially resolved transcriptomic studies. Nature Methods, 17(2), 193-200.
#'
#' Zhu, J., Sun, S., & Zhou, X. (2021). SPARK-X: non-parametric modeling enables
#' scalable and robust detection of spatial expression patterns for large spatial
#' transcriptomic studies. Genome Biology, 22(1), 184.
#'
#' @seealso
#' \code{\link{CreateSpatialXObject}}, \code{\link{ComputeACAT}},
#' \code{\link{ComputeSingleKernelMat}}
#' 
#' @importFrom stats setNames p.adjust
#' 
#' 
#' @name SpatialVG
#' @rdname SpatialVG
#' @export
#' 
SpatialVGC.default <- function(object,
                              cell.prop = NULL, ## cell type proportion
                              cell.type = NULL, ## cell type interested to test, test all if cell.type=NULL
                              location = NULL,
                              lib.size = NULL,
                              kernel.mat = NULL,
                              kernel.type = "gaussian",
                              kernel.param = NULL,
                              check.positive = FALSE,
                              fit.maxiter = 500, 
                              fit.tol = 1e-5,
                              fit.model = "poisson",
                              weights = NULL,
                              num.core = 1, 
                              verbose = FALSE) {
  
  ## number of cells and genes
  num_cell <- ncol(object)
  num_gene <- nrow(object)
  genes.use <- rownames(object)
  
  ## all cell type proportions, i.e., cell type proportion as covariates to remove
  if(is.null(cell.prop)){
    num_cov <- 0
  }else{
    cell.prop <- as.matrix(cell.prop)
    num_cov <- ncol(cell.prop)
  }## end fi
  
  if(is.null(cell.type)){
    cell.type <- colnames(cell.prop)
  }## end fi cell.type
  #################
  cat(paste("## ===== SPATIALVGC INPUT INFORMATION ====## \n"))
  cat(paste("## the model fitting: ", fit.model," model\n"))
  cat(paste("## number of total samples: ", ncol(object),"\n"))
  cat(paste("## number of total features: ", nrow(object),"\n"))
  cat(paste("## number of adjusted cell types: ", num_cov,"\n"))
  cat(paste("## number of cores: ", num.core,"\n"))
  cat(paste("## the kernel type: ", paste0(kernel.type,collapse = ","),"\n"))
  cat(paste("## the kernel parameters: ", paste0(kernel.param,collapse = ","),"\n"))
  cat(paste("## ===== END INFORMATION ==== \n"))
  cat("\n")
  ## due to we use the score tests, the kernel matrix will be calculated after the model fitting
  
  ## main functions
  if(fit.model == "poisson"){
    ##*************************************************##
    ##   Count-Based Spatial Model Under The Null      ##
    ##*************************************************##
    if(verbose) cat("# fitting count-based spatial model ... \n")
    if(is.null(lib.size)){
      lib.size <- Matrix::colSums(object) + 1; ## to avoid the zeros total counts
    }## end fi
    #=====================================
    gc();
  }else if(fit.model == "binomial"){
    #*************************************************#
    #   Binary-Based Spatial Model Under The Null     #
    #*************************************************#
    ##object@binary_counts <- as(object@counts>0, "dgCMatrix")
    object <- (object>0) + 0
    
    #class(binary_counts) <- "numeric"
    if(verbose) cat("# fitting binary-based spatial models ... \n")
  }## end fi fit.model
  
  ## generate the kernel matrices if does not provided by user
  if(is.null(kernel.mat)){
    ## check either kernel.type or kernel.param
    if(is.null(kernel.type) || is.null(kernel.param)){
      stop("SpatialVG::Please provide either 'kernel.type' or 'kernel.mat'")
    }## end fi
    
    if(length(kernel.type) != length(kernel.param)){
      stop("SpatialVG::Please provide same length of 'kernel.type' or 'kernel.param'")
    }## end fi
    ## number of the kernels
    num.kernel <- length(kernel.param)
    ED <- as.matrix(dist(location[ ,1:2]))
    ## compute the kernel matrix
    kernel_mat <- list()
    for (ikernel in 1:length(kernel.param)) {
      kernel_mat_each <- ComputeSingleKernelMat(ED=ED, kernel.type = kernel.type[ikernel], kernel.params = kernel.param[ikernel])
      kernel_mat <- c(kernel_mat, list(kernel_mat_each))
      rm(kernel_mat_each)
    }# end for ikernel
    rm(ED);gc();
    names(kernel_mat) <- names(kernel.param)
    
    # raw_kernel_mat <- ComputeSingleKernelMat(location, kernel.type = kernel.type, kernel.params = kernel.param)
    
  }else{## given the kernel matrix to fit
    if(!is.null(kernel.mat) & !is.list(kernel.mat)){stop("SpatialVGC::kernel.mat must be a list, please provide list type!")}
    
    num_kernel <- length(kernel.mat)
    if(is.null(names(kernel.mat))){
      names(kernel.mat) <- paste0("Ker", 1:num_kernel)
    }## end fi
    
  }## end fi kernel.mat
  #################### run main function with kernels
  res.final <- list()
  ## test each cell type at a time
  for(ict in 1:length(cell.type)){
    if(verbose) cat(paste0("# fitting null spatial models for cell type: ",cell.type[ict]," ... \n"))
    res_vc <- pbmcapply::pbmclapply(seq_len(num_gene), mc.cores = num.core, function(x){
      #for each condition get data as y_data
      tryCatch({suppressWarnings(
        model1 <- spark.null(object[x,], 
                             covariates = cell.prop[, -ict],
                             lib.size = lib.size,
                             fit.model = fit.model, 
                             fit.maxiter = fit.maxiter, 
                             fit.tol = fit.tol, 
                             verbose = verbose)
      ) 
      }, warning=function(w){ 
        print(w); return(model1);
      }, error=function(e){
        print(e); return(NULL);
      }, finally={
        #######
        return(model1)
      })
    })## end parallel
    
    names(res_vc) <- rownames(object)
    # rm(object);
    gc();
    ############## variance components testing
    pvalues <- NULL
    for(ikernel in 1:length(kernel.param)){
      if(verbose) {
        cat(paste0("# testing kernels of spatial models for cell type: ",cell.type[ict]," and kernel: ",names(kernel.param)[ikernel],"... \n"))
      }## end fi verbose
      kernel_mat_each <- sweep(kernel_mat[[ikernel]], MARGIN = 1, STATS = cell.prop[,ict], FUN = "*")
      kernel_mat_each <- sweep(kernel_mat_each, MARGIN = 2, STATS = cell.prop[,ict], FUN = "*")
      diag(kernel_mat_each) <- 1
      ## pre-defined kernels
      res.all.genes <- parallel::mclapply(seq_len(num_gene), mc.cores = num.core, 
                                          function(x){
                                            model1 <- res_vc[[x]]
                                            if(class(model1) != "try-error"){
                                              # res <- ComputeTestQuantRcpp_cov(model1$Y, 
                                              #                                 model1$Py, 
                                              #                                 model1$X, 
                                              #                                 as.matrix(kernel_mat_each)[model1$idx, model1$idx], 
                                              #                                 model1$D^2, 
                                              #                                 model1$theta)
                                              res <- FastTraceComputeTestQuantRcpp_cov(model1$Py, 
                                                                              model1$X,
                                                                              as.matrix(kernel_mat_each)[model1$idx, model1$idx], 
                                                                              model1$D^2, 
                                                                              model1$theta)
                                              newInfoM <- res$infoMp1
                                              ## calculate the scale parameter and degree of freedom
                                              ee_sw <- res$ee
                                              kk_sw <- newInfoM/(2.0*ee_sw)
                                              df_sw <- 2.0*ee_sw^2/(newInfoM)
                                              davies_sw <- pchisq(res$S0/kk_sw, df_sw, lower.tail = FALSE)
                                              if(verbose){cat(paste("SpatialVG::SW pvalue 1 = ", davies_sw,"\n"))}
                                            }else{
                                              davies_sw <- NA
                                            }## end fi
                                            
                                            #######
                                            return(data.frame(davies_sw, res$S0, newInfoM))
                                          })## end parallel genes
      rm(kernel_mat_each);gc();
      res.int <- matrix(unlist(res.all.genes), ncol = 3, byrow = TRUE)
      colnames(res.int) <- paste0(c("pvalue.","stat.","var."),names(kernel.param)[ikernel])
      rownames(res.int) <- names(res_vc)
      pvalues <- cbind(pvalues, res.int)
      rm(res.int);rm(res.all.genes);gc();
    }## end for ikernel
    
    res <- setNames(split(pvalues[,grepl("pvalue.",colnames(pvalues)),drop=FALSE], 
                          seq(nrow(pvalues))), rownames(pvalues))
    ## integrate 10 p-values into one with weights, each gene at a time
    combined_pvalue <- unlist(lapply(res, ComputeACAT, Weights = weights))
    combined_pvalue_final <- data.frame(pvalues, 
                                        combined_pvalue = combined_pvalue, 
                                        padj_BY = p.adjust(combined_pvalue, method = "BY"))
    ## return results
    res.final[[ict]] <- combined_pvalue_final[order(combined_pvalue_final$combined_pvalue, decreasing = FALSE), ]
  }## end fi cell types
  names(res.final) <- cell.type
  return(res.final)
}## end function 



#'
#' Fitting the binary or count spatial model to perform spatially-aware expression analysis for spatial data
#' 
#' @rdname SpatialVG
#' @export
#' 
SpatialVG.default <- function(object,
					svg.method = NULL,
					covariates = NULL,
					location = NULL,
					lib.size = NULL,
					kernel.mat = NULL,
					kernel.type = "gaussian",
					kernel.param = NULL,
					check.positive = FALSE,
					fit.maxiter = 500, 
					fit.tol = 1e-5,
					fit.model = "poisson",
					weights = NULL,
					num.core = 1, 
					verbose = FALSE, ...) {
	
	## number of cells and genes
	num_cell <- ncol(object)
	num_gene <- nrow(object)
	genes.use <- rownames(object)
	
	## covariates, i.e., confounding or batch effects
	if(is.null(covariates)){
		num_cov <- 0
	}else{
		covariates <- as.matrix(covariates)
		num_cov <- ncol(covariates)
	}## end fi
	
	if(!is.null(kernel.mat) & !is.list(kernel.mat)){stop("SpatialVG::kernel.mat must be a list, please provide list type!")}

	#################
	cat(paste("## ===== SPATIALVG INPUT INFORMATION ====## \n"))
	cat(paste("## the model fitting: ", fit.model," model\n"))
	cat(paste("## number of total samples: ", ncol(object),"\n"))
	cat(paste("## number of total features: ", nrow(object),"\n"))
	cat(paste("## number of adjusted covariates: ", num_cov,"\n"))
	cat(paste("## number of cores: ", num.core,"\n"))
	cat(paste("## the kernel type: ", paste0(kernel.type,collapse = ","),"\n"))
	cat(paste("## the kernel parameters: ", paste0(kernel.param,collapse = ","),"\n"))
	cat(paste("## ===== END INFORMATION ==== \n"))
	cat("\n")
	## due to we use the score tests, the kernel matrix will be calculated after the model fitting
	
	## main functions
	if(fit.model == "poisson"){
	##*************************************************##
	##   Count-Based Spatial Model Under The Null      ##
	##*************************************************##
	  if(verbose) cat("# fitting count-based spatial model ... \n")
		if(is.null(lib.size)){
			lib.size <- Matrix::colSums(object) + 1; ## to avoid the zeros total counts
		}## end fi
		#=====================================
	  gc();
		#res_vc <- parallel::mclapply(seq_len(num_gene), mc.cores = num.core, function(x){
		res_vc <- pbmcapply::pbmclapply(seq_len(num_gene), mc.cores = num.core, function(x){
		  #for each condition get data as y_data
		  tryCatch({suppressWarnings(
		    model1 <- spark.null(object[x,], 
		                         covariates = covariates,
		                         lib.size = lib.size,
		                         fit.model = fit.model, 
		                         fit.maxiter = fit.maxiter, 
		                         fit.tol = fit.tol, 
		                         verbose = verbose)
		      ) 
		    }, warning=function(w){ 
		      print(w); return(model1);
		    }, error=function(e){
		      print(e); return(NULL);
		    }, finally={
		      #######
		      return(model1)
		    }
		  )
		})## end parallel
		rm(object);
		gc();
		
		############## variance components testing
		if(verbose) cat("# testing kernels of spatial models ... \n")
		## generate the kernel matrices if does not provided by user
		if(is.null(kernel.mat)){
		  ## check either kernel.type or kernel.param
		  if(is.null(kernel.type) || is.null(kernel.param)){
		    stop("SpatialVG::Please provide either 'kernel.type' or 'kernel.mat'")
		  }## end fi
		  
		  if(length(kernel.type) != length(kernel.param)){
		    stop("SpatialVG::Please provide same length of 'kernel.type' or 'kernel.param'")
		  }## end fi
		  ## number of the kernels
		  num.kernel <- length(kernel.param)

			## compute p-values for all genes
			Y <- do.call(cbind,lapply(res_vc, function(imodel){return(imodel$Y)}))
			Dsq <- do.call(cbind,lapply(res_vc, function(imodel){return(imodel$D^2)}))
			Theta <- do.call(cbind,lapply(res_vc, function(imodel){return(imodel$theta)}))
			index <- do.call(cbind,lapply(res_vc, function(imodel){return(imodel$idx)}))
			rm(res_vc); 
			gc();
			## modified by sun, 2023-7-19 09:08:41, it looks like more cores too slow.
			## modified by sun, 2025-9-25 20:15:47
			# num.core <- 1
			fx_test <- function(i){
			  res <- fastComputeQuantitiesWithoutCVT(Y, 
	                                            location, 
	                                            as.numeric(kernel.param[i]), 
	                                            Dsq, 
	                                            Theta, 
	                                            kernel.type[i])
			  ## calculate the scale parameter and degree of freedom
			  newInfoM <- res$infoMp1
			  ee_sw <- res$ee
			  kk_sw <- newInfoM/(2.0*ee_sw)
			  df_sw <- 2.0*ee_sw^2/(newInfoM)
			  davies_sw <- pchisq(res$S0/kk_sw, df_sw, lower.tail = FALSE)
			  if(verbose) {cat(paste("SpatialVG::SW top 5 pvalues = ", davies_sw[1:5],"\n"))}
			  return(data.frame(davies_sw, res$S0, newInfoM))
			}
			res_davies <- parallel::mclapply(seq_len(num.kernel), fx_test, mc.cores = 1)
			
			## format transformation
			pvalues_datframe <- do.call(cbind, res_davies)
			colnames(pvalues_datframe) <- paste0(c("pvalue.","stats.", "var."), rep(names(kernel.param),each=3))
			pvalues <- data.frame(pvalues_datframe, row.names = make.names(genes.use,unique = TRUE))
			
			res <- setNames(split(pvalues[,grepl("pvalue.",colnames(pvalues)),drop=FALSE], 
			                      seq(nrow(pvalues))), rownames(pvalues))
			
		}else{##============== the kernel provided by users, testing one by one
		  if(!is.list(kernel.mat)){
		    stop("SpatialVG::The 'kernel.mat' must be a list.")
		  }## end fi
		  num_kernel <- length(kernel.mat)
		  if(is.null(names(kernel.mat))){
		    names(kernel.mat) <- paste0("Ker", 1:num_kernel)
		  }## end fi
			
			pvalues <- NULL
			for(ikernel in 1:num_kernel){
				## pre-defined kernels
				cat(paste0("## testing pre-defined kernel: ",ikernel,"...\n"))
				res.all.genes <- parallel::mclapply(seq_len(num_gene), mc.cores = num.core, 
					function(x){
					model1 <- res_vc[[x]]
					if(class(model1) != "try-error"){
						if(num_cov > 0){
							res <- ComputeTestQuantRcpp_cov(model1$Y, 
							                                 model1$Py, 
							                                 model1$X, 
							                                 as.matrix(kernel.mat)[model1$idx, model1$idx], 
							                                 model1$D^2, 
							                                 model1$theta)
						}else{
							res <- ComputeTestQuantRcpp_nocov(model1$Y, 
							                                   model1$Py, 
							                                   as.matrix(kernel.mat)[model1$idx, model1$idx], 
							                                   model1$D^2, 
							                                   model1$theta)
						}## end fi
						newInfoM <- res$infoMp1
						## calculate the scale parameter and degree of freedom
						ee_sw <- res$ee
						kk_sw <- newInfoM/(2.0*ee_sw)
						df_sw <- 2.0*ee_sw^2/(newInfoM)
						davies_sw <- pchisq(res$S0/kk_sw, df_sw, lower.tail = FALSE)
						if(verbose){cat(paste("SpatialVG::SW pvalue 1 = ", davies_sw,"\n"))}
					}else{
						davies_sw <- NA
					}## end fi
						
					#######
					return(data.frame(davies_sw, res$S0, newInfoM))
				})## end parallel genes
				pvalues <- cbind(pvalues, matrix(unlist(res.all.genes), ncol = 3, byrow = TRUE) )
			}## end for ikernel
			colnames(pvalues) <- names(kernel.mat)
			rownames(pvalues) <- make.names(genes.use,unique = TRUE)
		}## end fi
	}else if(fit.model == "binomial"){
	#*************************************************#
	#   Binary-Based Spatial Model Under The Null     #
	#*************************************************#
		##object@binary_counts <- as(object@counts>0, "dgCMatrix")
		object <- (object>0) + 0
		
		#class(binary_counts) <- "numeric"
		if(verbose) cat("# fitting binary-based spatial models ... \n")
		##=====================================
		res_vc <- pbmcapply::pbmclapply(seq_len(num_gene), mc.cores = num.core, function(x){
		  #for each condition get data as y_data
		  tryCatch({suppressWarnings(
		    model1 <- spark.null(object[x,], 
		                         covariates = covariates,
		                         fit.model = fit.model, 
		                         fit.maxiter = fit.maxiter, 
		                         fit.tol = fit.tol, 
		                         verbose = verbose)
		  ) 
		  }, warning=function(w){ 
		    print(w); return(model1);
		  }, error=function(e){
		    print(e); return(NULL);
		  }, finally={
		    #######
		    return(model1)
		  })
		})## end parallel
		rm(object);
		gc();
		
		
		############## testing
		if(verbose) cat("# testing kernels of spatial models ... \n")
		## generate the kernel matrices if does not provided by user
		if(is.null(kernel.mat)){
		  ## check either kernel.type or kernel.param
		  if(is.null(kernel.type) || is.null(kernel.param)){
		    stop("SpatialVG::Please provide either 'kernel.type' or 'kernel.mat'")
		  }## end fi
		  
		  if(length(kernel.type) != length(kernel.param)){
		    stop("SpatialVG::Please provide same length of 'kernel.type' or 'kernel.param'")
		  }## end fi
		  ## number of the kernels
		  num.kernel <- length(kernel.param)
		  
			## compute p-values for all genes
		  Y <- do.call(cbind,lapply(res_vc, function(imodel){return(imodel$Y)}))
		  Dsq <- do.call(cbind,lapply(res_vc, function(imodel){return(imodel$D^2)}))
		  Theta <- do.call(cbind,lapply(res_vc, function(imodel){return(imodel$theta)}))
		  index <- do.call(cbind,lapply(res_vc, function(imodel){return(imodel$idx)}))
		  rm(res_vc); 
		  gc();
		  ## modified by sun, 2023-7-19 09:08:41, it looks like more cores too slow. 
		  ##num.core <- 1
		  ## each kernel at a time
		  fx_test <- function(i){
		    res <- fastComputeQuantitiesWithoutCVT(Y, 
		                                           location, 
		                                           as.numeric(kernel.param[i]), 
		                                           Dsq, 
		                                           Theta, 
		                                           kernel.type[i])
		    ## calculate the scale parameter and degree of freedom
		    newInfoM <- res$infoMp1
		    ee_sw <- res$ee
		    kk_sw <- newInfoM/(2.0*ee_sw)
		    df_sw <- 2.0*ee_sw^2/(newInfoM)
		    davies_sw <- pchisq(res$S0/kk_sw, df_sw, lower.tail = FALSE)
		    if(verbose) {cat(paste("SpatialVG::SW top 5 pvalues = ", davies_sw[1:5],"\n"))}
		    return(data.frame(davies_sw, res$S0, newInfoM))
		  }
		  res_davies <- parallel::mclapply(seq_len(num.kernel), fx_test, mc.cores = 1)
		  
		  ## format transformation
		  pvalues <- do.call(cbind, res_davies)
		  colnames(pvalues) <- paste0(c("pvalue.","stats.", "var."), rep(names(kernel.param),each=3))
      pvalues <- data.frame(pvalues, 
                            row.names = make.names(genes.use,unique = TRUE))
      
      ## split p-values into different group for combining
      res <- setNames(split(pvalues[,grepl("pvalue.",colnames(pvalues)),drop=FALSE], 
                            seq(nrow(pvalues))), rownames(pvalues))
      
		}else{##============== the kernel provided by users, testing one by one
		  if(!is.list(kernel.mat)){
		    stop("SpatialVG::The 'kernel.mat' must be a list.")
		  }## end fi
		  num_kernel <- length(kernel.mat)
		  if(is.null(names(kernel.mat))){
		    names(kernel.mat) <- paste0("Ker", 1:num_kernel)
		  }## end fi
			
			pvalues <- NULL
			for(ikernel in 1:num_kernel){
				## pre-defined kernels
				cat(paste0("## testing pre-defined kernel: ",ikernel,"...\n"))
				res.all.genes <- parallel::mclapply(seq_len(num_gene), mc.cores = num.core, 
					function(x){
					model1 <- res_vc[[x]]
					if(class(model1) != "try-error"){
						if(num_cov > 0){
							res <- ComputeTestQuantRcpp_cov(model1$Y, 
							                                 model1$Py, 
							                                 model1$X, 
							                                 as.matrix(kernel.mat)[model1$idx, model1$idx], 
							                                 model1$D^2, 
							                                 model1$theta)
						}else{
							res <- ComputeTestQuantRcpp_nocov(model1$Y, 
							                                   model1$Py, 
							                                   as.matrix(kernel.mat)[model1$idx, model1$idx], 
							                                   model1$D^2, 
							                                   model1$theta)
						}## end fi
						newInfoM <- res$infoMp1
						## calculate the scale parameter and degree of freedom
						ee_sw <- res$ee
						kk_sw <- newInfoM/(2.0*ee_sw)
						df_sw <- 2.0*ee_sw^2/(newInfoM)
						davies_sw <- pchisq(res$S0/kk_sw, df_sw, lower.tail = FALSE)
						if(verbose){cat(paste("SpatialVG::SW pvalue 1 = ", davies_sw,"\n"))}
					}else{
						davies_sw <- NA
					}## end fi
						
					#######
					return(data.frame(davies_sw, res$S0, newInfoM))
				})## end parallel genes
				pvalues <- cbind(pvalues, matrix(unlist(res.all.genes), ncol = 3, byrow = TRUE) )
			}## end for ikernel
			colnames(pvalues) <- names(kernel.mat)
			rownames(pvalues) <- make.names(genes.use,unique = TRUE)
		}## end fi kernel.mat is provided

	}else if(fit.model == "gaussian"){
	#*********************************************#
	#    Normalized Count-Based Spatial Model     #
	#*********************************************#
		cat("# fitting gaussian version spatial model ... \n")
		## covariates are required in the downstream steps
		## if null add a column vector with 1
		if(is.null(covariates)){
			covariates <- as.matrix(rep(1, num_cell))
		}else if(!is.matrix(covariates)){
			## if null add a column vector with 1
			covariates <- as.matrix(covariates)
		}## end fi
		num_cov <- ncol(covariates)
		zeros_threshold <- 10e-5
		
		## transpose gene expression matrix
		location <- scale(location)
		#norm_counts <- t(object)
		
		if(is.null(kernel.mat)){
		  ## check either kernel.type or kernel.param
		  if(is.null(kernel.type) || is.null(kernel.param)){
		    stop("SpatialVG::Please provide either 'kernel.type' or 'kernel.mat'")
		  }## end fi
		  
		  if(length(kernel.type) != length(kernel.param)){
		    stop("SpatialVG::Please provide same length of 'kernel.type' or 'kernel.param'")
		  }## end fi
		  ## number of the kernels
		  num.kernel <- length(kernel.param)
		}## end fi
		
		## loop to check each kernels, kernel_mat
		ED <- ED_cpp(as.matrix(location))
		ED <- ED^2
		## calculate Moore-Penrose pseudoinverse for covariates
		Xdagger <- pracma::pinv(covariates)
		kernel.names <- names(kernel.param)
		pvalues <- matrix(0, nrow = num_gene, ncol=length(kernel.param))
		res_davies <- parallel::mclapply(seq_len(kernel.param), mc.cores = num.core, function(x){
		#for(ikernel in 1:length(kernel.param)){
		cat(paste0("## compute the pvalues with kernel ",kernel.names[x]," ... \n"))
		  kernel_mat <- ComputeSingleKernelMat(ED = ED, 
		                                    kernel.params = kernel.param[x], 
		                                    kernel.type = kernel.names[x],
		                                    check.positive = check.positive)
		  ## spark gaussian version
		  res <- spark.g(norm_counts = object, 
		                 Kmat = kernel_mat, 
		                 covariates = covariates, 
		                 Xdagger = Xdagger, 
		                 verbose = verbose)
		  return(res[,1])
		})
		pvalues <- do.call(cbind, res_davies)
		rownames(pvalues) <- make.names(genes.use,unique = TRUE)
		colnames(pvalues) <- names(kernel.param)
		res <- setNames(split(pvalues, seq(nrow(pvalues))), rownames(pvalues))
	}else if(fit.model == "nonparametric"){
	  #**********************************************************#
	  #    Normalized Non-parametric (SPARK-X) Spatial Model     #
	  #**********************************************************#
	  cat("# fitting non-parametric (sparkx) version spatial model ... \n")
	  ## covariates are required in the downstream steps
	  ## if null add a column vector with 1
	  if(is.null(covariates)){
	    covariates <- NULL
	  }else if(!is.matrix(covariates)){
	    ## if null add a column vector with 1
	    covariates <- as.matrix(covariates)
	  }## end fi
  
	  ## transpose gene expression matrix
	  location <- scale(location)
 
	  ## check either kernel.type or kernel.param
	  if(is.null(kernel.type) || is.null(kernel.param)){
	    stop("SpatialVG::Please provide either 'kernel.type' or 'kernel.mat'")
	  }## end fi
	  
	  if(length(kernel.type) != length(kernel.param)){
	    stop("SpatialVG::Please provide same length of 'kernel.type' or 'kernel.param'")
	  }## end fi
	  ## number of the kernels
	  num.kernel <- length(kernel.param)
	  kernel.names <- names(kernel.param)
	  pvalues <- matrix(0, nrow = num_gene, ncol=length(kernel.names))
	  res_davies <- parallel::mclapply(seq_len(kernel.param), mc.cores = num.core, function(x){
	    cat(paste0("## compute the pvalues with kernel ",kernel.names[ikernel]," ... \n"))
	    final_location <- apply(location, 2, 
	                            transloc_func, 
	                            lker = kernel.param[ikernel], 
	                            transfunc = kernel.names[ikernel])
	    ## sparkx non-parametric version 
	    res <- spark.x(counts = object, 
	                   infomat = final_location, 
	                   X_mat = covariates, 
	                   mc_cores = num.core, 
	                   verbose = verbose)
	    return(res[,2])
	  })
	  
	  pvalues <- do.call(cbind, res_davies)
	  rownames(pvalues) <- make.names(genes.use,unique = TRUE)
	  colnames(pvalues) <- names(kernel.param)
	  res <- setNames(split(pvalues, seq(nrow(pvalues))), rownames(pvalues))
	}## end fi
	
	## integrate 10 p-values into one with weights, each gene at a time
	combined_pvalue <- unlist(lapply(res, ComputeACAT, Weights = weights))
	combined_pvalue_final <- data.frame(pvalues, 
                                      combined_pvalue = combined_pvalue, 
                                      padj_BY = p.adjust(combined_pvalue, method = "BY"))
	## return results
	combined_pvalue_final <- combined_pvalue_final[order(combined_pvalue_final$combined_pvalue, decreasing = FALSE), ]
	return(combined_pvalue_final)
}## end function 



#' SpatialX Assay for spatially variable gene identification
#' @rdname SpatialVG
#' @export
#' 
SpatialVG.Assay <- function(object,
      	slot.use = "counts",
      	location = NULL,
      	cell.type = NULL,
      	features = NULL,
      	svg.method = "spark",
      	top.features = 2000,
      	svg.param = list(covariates = NULL,
      				lib.size = NULL,
      				fit.maxiter = 100, 
      				fit.tol = 1e-5,
      				fit.model = "poisson"),
      	kernel.mat = NULL, 
      	kernel.param = list(band.with = 2.0,
      					kernel.type = "gaussian",
      					check.positive = FALSE,
      					weights = NULL),
      	num.core = 1,
      	verbose = TRUE, ...) {
  
  ## spatial location
  if(is.null(location)){
    location <- GetAssayData(object = object, slot = "location");
    if(!is.matrix(location)){
      location <- as.matrix(location)
    }## end fi
  }## end fi
  
  if(is.null(cell.type)){
    cell.prop <- GetAssayData(object = object, slot = "cell.prop");
    cell.type <- colnames(cell.prop)
  }## end fi is.null
  
  ## parameter settings check
  if(svg.method == "spark"){## nature methods' paper, SPARK
    svg.param$fit.model <- "poisson"
    ## 10 kernels
    kernel.param$band.with <- ComputeKernelParamLessMem(location)$kparam[3:7]
    kernel.param$band.with <- rep(kernel.param$band.with, times=2)
    names(kernel.param$band.with) <- c(paste0("GSP", seq_len(5)), paste0("COS", seq_len(5)))
    kernel.param$kernel.type <- c(rep("gaussian", 5), rep("cosine",5))
    if(slot.use != "counts"){
      stop(paste0("User provide method ", svg.method," requires 'slot.use = counts' \n"))
    }## end fi
  }else if(svg.method == "sparkx"){## genome biology's paper, SPARK-X
    svg.param$fit.model <- "nonparametric"
    ## 10 kernels
    kernel.param$band.with <- ComputeKernelParamLessMem(location)$kparam[3:7]
    kernel.param$band.with <- rep(kernel.param$band.with, times=2)
    names(kernel.param$band.with) <- c(paste0("GSP", seq_len(5)), paste0("COS", seq_len(5)))
    kernel.param$kernel.type <- c(rep("gaussian", 5), rep("cosine",5))
    if( !(slot.use %in% c("data","scale.data"))){
      stop(paste0("User provide method ", svg.method," requires 'slot.use = data/scale.data' \n"))
    }## end fi
  }else if(svg.method == "sparkb"){## optimal for binarized counts data or highly-sparse data
    svg.param$fit.model <- "binomial"
    ## 10 kernels
    kernel.param$band.with <- ComputeKernelParamLessMem(location)$kparam[3:7]
    kernel.param$band.with <- rep(kernel.param$band.with, times=2)
    names(kernel.param$band.with) <- c(paste0("GSP", seq_len(5)), paste0("COS", seq_len(5)))
    kernel.param$kernel.type <- c(rep("gaussian", 5), rep("cosine",5))
    if(slot.use != "counts"){
      stop(paste0("User provide method ", svg.method," requires 'slot.use = counts' \n"))
    }## end fi
  }else if(svg.method == "csvg"){## cell-type specific svgs
    svg.param$fit.model <- "poisson"
    ## 10 kernel
    kernel.param$band.with <- ComputeKernelParamLessMem(location)$kparam[3:7]
    kernel.param$band.with <- rep(kernel.param$band.with, times=2)
    names(kernel.param$band.with) <- c(paste0("gaussian", seq_len(5)), paste0("matern", seq_len(5)))
    kernel.param$kernel.type <- c(rep("gaussian", 5), rep("matern",5))
    if(slot.use != "counts"){
      stop(paste0("User provide method ", svg.method," requires 'slot.use = counts' \n"))
    }## end fi
    
  }else if(svg.method == "dsvg"){## domain specific svgs
    svg.param$fit.model <- "poisson"
    ## 10 kernels
    
    
  }else if(svg.method == "isvg"){## interaction svgs
    svg.param$fit.model <- "poisson"
    ## 10 kernels
    
    
  }else{
    if(is.null(svg.param$fit.model)){
      stop("SpatialVG.Assay::The 'fit.model' must be provided by user.")
    }## end fi
    if(is.null(kernel.param$band.with) || is.null(kernel.param$kernel.type)){
      stop("SpatialVG.Assay::The 'fit.model' must be provided by user.")
    }
  }## end fi
  
  ## normalized counts, vst normalization method
  if(svg.param$fit.model == "gaussian"){
    if( !(slot.use %in% c("data","scale.data"))){
      stop(paste0("SpatialVG.Assay::User provide method ", svg.method," requires 'slot.use = data/scale.data' \n"))
    }## end fi
  }else if(svg.param$fit.model == "poisson"){
    if(slot.use != "counts"){
      stop(paste0("SpatialVG.Assay::User provide method ", svg.method," requires 'slot.use = counts' \n"))
    }## end fi
  }else if(svg.param$fit.model == "binomial"){
    if(slot.use != "counts"){
      stop(paste0("SpatialVG.Assay::User provide method ", svg.method," requires 'slot.use = counts' \n"))
    }## end fi
  }else if(svg.param$fit.model == "nonparameteric"){
    if( !(slot.use %in% c("data","scale.data"))){
      stop(paste0("SpatialVG.Assay::User provide method ", svg.method," requires 'slot.use = data/scale.data' \n"))
    }## end fi
  }## end fi
  
  ## parameter settings
  
  ## data use
	data.use <- GetAssayData(object = object, slot = slot.use)
	features <- features %||% rownames(x = data.use)
	
	
	if(length(data.use) == 0){
		stop(paste0("SpatialVG.Assay::::Please provide the slot ", slot.use, " data before running SpatialVG function. \n"))
	}## end fi
	if(svg.method == "csvg"){
	  new.data <- SpatialVGC.default(object = data.use[features, , drop=FALSE],
	                        cell.prop = GetAssayData(object = object, slot = "cell.prop"),
	                        cell.type = cell.type,
	                        location = location,
	                        lib.size = svg.param$lib.size,
	                        kernel.mat = kernel.mat,
	                        kernel.type = kernel.param$kernel.type,
	                        kernel.param = kernel.param$band.with,
	                        check.positive = kernel.param$check.positive,
	                        weights = kernel.param$weights,
	                        fit.maxiter = svg.param$fit.maxiter, 
	                        fit.tol = svg.param$fit.tol,
	                        fit.model = svg.param$fit.model,
	                        num.core = num.core,
	                        verbose = verbose, ...)
	  
	  
	  ## store the scaled data in the slot
	  object <- SetAssayData(object = object, slot = 'csvg', new.data = new.data)
	}else if(svg.method == "dsvg"){
	  
	}else if(svg.method == "isvg"){
	  
	}else if(svg.method %in%c("spark","sparkb","sparkx")){
	  ## run default SpatialVG, assume new.data has been ordered by combined p-values
	  new.data <- SpatialVG(object = data.use[features, , drop=FALSE],
	                        svg.method = svg.method,
	                        covariates = NULL,
	                        location = location,
	                        lib.size = svg.param$lib.size,
	                        kernel.mat = kernel.mat,
	                        kernel.type = kernel.param$kernel.type,
	                        kernel.param = kernel.param$band.with,
	                        check.positive = kernel.param$check.positive,
	                        weights = kernel.param$weights,
	                        fit.maxiter = svg.param$fit.maxiter, 
	                        fit.tol = svg.param$fit.tol,
	                        fit.model = svg.param$fit.model,
	                        num.core = num.core,
	                        verbose = verbose, ...)
	  
	  
	  ## store the scaled data in the slot
	  object <- SetAssayData(object = object, slot = 'svg', new.data = new.data)
	  ## store top number of features in svg slot
	  object <- SetAssayData(object = object, slot = 'sv.genes', new.data = rownames(new.data)[1:min(top.features, nrow(new.data))])
	}## end fi svg.method
	
	
}## end func



#' SpatialX for spatially variable gene identification
#' @rdname SpatialVG
#' @export
#' 
SpatialVG.SpatialX <- function(object, 
                                assay = NULL,
                                slot.use = "counts",
                                features = NULL,
                                svg.method = "spark",
                                cell.type = NULL,
                                top.features = 2000,
                                svg.param = list(covariates = NULL,
                                			lib.size = NULL,
                                			fit.maxiter = 100, 
                                			fit.tol = 1e-5,
                                			fit.model = "poisson"),
                                kernel.mat = NULL, 
                                kernel.param = list(band.with = 2.0,
                                				kernel.type = "gaussian",
                                				check.positive = FALSE,
                                				weights = NULL),
                                num.core = 1,
                                verbose = TRUE, ...) {
	
	## parallel parameter setting
	if(num.core == 1){
		if(slot(object, name="num.core") > 1) {num.core <- slot(object,name = "num.core")}
	}## end fi
	
	## assays
	assay <- assay %||% DefaultAssay(object = object)
	assay.data <- GetAssay(object = object, assay = assay)

	## assay function
	assay.data <- SpatialVG(object = assay.data,
					slot.use = slot.use,
					features =  features,
					svg.method = svg.method,
					cell.type = cell.type,
					top.features = top.features,
					svg.param = svg.param,
					kernel.mat = kernel.mat,
					kernel.param = kernel.param,
					num.core = num.core,
					verbose = verbose, ... )

	## store back the treated data
	object@assays[[assay]] <- assay.data
	return(object)
}## end func

#' support Seurat object
#' @rdname SpatialVG
#' @export
#' 
SpatialVG.Seurat <- function(object, 
                               assay = NULL,
                               slot.use = "counts",
                               features = NULL,
                               location = NULL,
                               svg.method = "spark",
                               top.features = 2000,
                               svg.param = list(covariates = NULL,
                                                lib.size = NULL,
                                                fit.maxiter = 100, 
                                                fit.tol = 1e-5,
                                                fit.model = "poisson"),
                               kernel.mat = NULL, 
                               kernel.param = list(band.with = 2.0,
                                                   kernel.type = "gaussian",
                                                   check.positive = FALSE,
                                                   weights = NULL),
                               num.core = 1,
                               verbose = TRUE, ...) {
  
  ## version of object
  cat(paste0("## The version of Seurat object is ", Seurat::Version(object = object)," ... \n"))
  ## 
  ## assays
  assay.data <- new(Class = 'Assay', 
                    counts = Seurat::GetAssayData(object, slot="counts"), 
                    location = as.data.frame(Seurat::GetTissueCoordinates(object)),
                    scale.data = new(Class = 'matrix'),
                    meta.features = object@meta.data,
                    misc = list())
  
  ## check normalized counts, vst normalization method
  if(svg.param$fit.model == "gaussian"){ slot.use <- "scale.data" }## end fi
  
  assay.data <- SpatialVG(object = assay.data,
                          slot.use = slot.use,
                          features =  features,
                          svg.method = svg.method,
                          top.features = top.features,
                          svg.param = svg.param,
                          kernel.mat = kernel.mat,
                          kernel.param = kernel.param,
                          num.core = num.core,
                          verbose = verbose, ... )
  
  ## store back the treated data
  ## create new SpatialX object
  object <- new(Class = 'SpatialX',
                assays = list('Spatial' = assay.data),
                active.assay = "Spatial",
                project.name = "SVG")
  return(object)
}##

#' Support SpatialExperiment object
#' @rdname SpatialVG
#' @export
#' 
SpatialVG.SpatialExperiment <- function(object,
                               slot.use = "counts",
                               features = NULL,
                               location = NULL,
                               svg.method = "spark",
                               top.features = 2000,
                               svg.param = list(covariates = NULL,
                                                lib.size = NULL,
                                                fit.maxiter = 100, 
                                                fit.tol = 1e-5,
                                                fit.model = "poisson"),
                               kernel.mat = NULL, 
                               kernel.param = list(band.with = 2.0,
                                                   kernel.type = "gaussian",
                                                   check.positive = FALSE,
                                                   weights = NULL),
                               num.core = 1,
                               verbose = TRUE, ...) {
  
  ## assays
  
  ## Initialize meta.features
  #y <- SpatialExperiment::assays(object)[[assay]]
  # coords <- SpatialExperiment::spatialCoords(object)
  init.meta.features <- data.frame(row.names = rownames(x = object))
  assay.data <- new(Class = 'Assay', 
               counts = SingleCellExperiment::counts(object = object), 
               data = SingleCellExperiment::logcounts(object = object),
               location = as.data.frame(SpatialExperiment::spatialCoords(object)),
               scale.data = new(Class = 'matrix'),
               meta.features = init.meta.features,
               misc = list())
  ## check normalized counts, vst normalization method
  if(svg.param$fit.model == "gaussian"){ 
    slot.use <- "scale.data"
    assay.data <- SpatialTF(assay.data, scale.method = "vst", verbose = TRUE)
  }## end fi
  
  assay.data <- SpatialVG(object = assay.data,
                          slot.use = slot.use,
                          features =  features,
                          svg.method = svg.method,
                          top.features = top.features,
                          svg.param = svg.param,
                          kernel.mat = kernel.mat,
                          kernel.param = kernel.param,
                          num.core = num.core,
                          verbose = verbose, ... )
  
  ## store back the treated data
  ## create new SpatialX object
  object <- new(Class = 'SpatialX',
                assays = list('Spatial' = assay.data),
                active.assay = "Spatial",
                project.name = "SVG")
  #object@assays[[assay]] <- assay.data
  return(object)
}##


#' Differential expression analysis for SRT
#' @rdname SpatialVG
#' @export
#' 
SpatialVG <- function(object, ...) {
	UseMethod(generic = "SpatialVG", object = object)
}## end func



##############################################################################################

#' Fit null models for spatial pattern detection
#'
#' This function fits null models (without spatial random effects) for individual
#' genes as an initialization step for spatial pattern detection. It prepares
#' the baseline models that will later be extended with spatial components
#' in the full spatial analysis pipeline.
#'
#' @param y Gene expression vector for a single gene. For Poisson models, this
#'         should be raw counts; for binomial models, this should be binary
#'         (0/1) expression indicators.
#' @param covariates Optional matrix of covariates to adjust for in the null
#'                  model. Can include technical covariates like batch effects
#'                  or biological covariates like cell cycle scores.
#' @param lib.size Library size (total counts) for each cell. Required for
#'                Poisson models to account for sequencing depth variation.
#'                Default is NULL.
#' @param fit.model Statistical model family to use. Options are "poisson" for
#'                 count data or "binomial" for binary data. Default is "poisson".
#' @param fit.maxiter Maximum number of iterations for model fitting. Default is 100.
#' @param fit.tol Tolerance threshold for model convergence. Default is 1e-5.
#' @param verbose Logical indicating whether to print progress messages and
#'               fitting information. Default is FALSE.
#'
#' @return Returns a list with the following components:
#'   \itemize{
#'     \item Model components from \code{\link{spark.fit}} including:
#'       \itemize{
#'         \item \code{theta}: Estimated variance components
#'         \item \code{coefficients}: Fixed effects coefficients
#'         \item \code{linear.predictors}: Linear predictors
#'         \item \code{fitted.values}: Fitted values
#'         \item \code{Y}: Working response vector
#'         \item \code{residuals}: Response residuals
#'         \item \code{Py}: Projected working response
#'         \item \code{X}: Design matrix
#'         \item \code{D}: Diagonal weights matrix
#'         \item \code{converged}: Convergence status
#'       }
#'     \item \code{times}: Computation time in seconds
#'     \item \code{idx}: Indices of non-missing observations
#'   }
#'
#' @details
#' This function serves as the initialization step in the spatial analysis
#' pipeline by fitting null models that don't include spatial random effects.
#' These null models provide:
#'
#' \itemize{
#'   \item Baseline parameter estimates for fixed effects
#'   \item Initial values for variance components
#'   \item Working responses and weights for spatial model fitting
#'   \item Identification of non-missing observations
#' }
#'
#' The function handles two main data types:
#' \describe{
#'   \item{Poisson models}{For count data with library size offset to account
#'         for sequencing depth variation. Requires the \code{lib.size} parameter.}
#'   \item{Binomial models}{For binary data (expressed/not expressed) without
#'         library size adjustment.}
#' }
#'
#' The fitting process:
#' \enumerate{
#'   \item Constructs the appropriate formula based on covariates and model family
#'   \item Fits an initial GLM using \code{\link{glm}}
#'   \item Refines the fit using \code{\link{spark.fit}} for robust estimation
#'   \item Records computation time and observation indices
#'   \item Returns the complete null model object for spatial extension
#' }
#'
#' @examples
#' \dontrun{
#' # Poisson model example
#' set.seed(123)
#' n_cells <- 100
#' 
#' # Generate Poisson count data
#' y_pois <- rpois(n_cells, lambda = 10)
#' lib_size <- rpois(n_cells, lambda = 1000)  # Library sizes
#' covariates <- matrix(rnorm(n_cells * 2), ncol = 2)  # Two covariates
#' 
#' # Fit null model for Poisson data
#' null_model_pois <- spark.null(
#'   y = y_pois,
#'   covariates = covariates,
#'   lib.size = lib_size,
#'   fit.model = "poisson",
#'   verbose = TRUE
#' )
#' 
#' # Binomial model example
#' y_bin <- rbinom(n_cells, 1, 0.3)  # Binary expression
#' 
#' # Fit null model for binary data
#' null_model_bin <- spark.null(
#'   y = y_bin,
#'   covariates = covariates,
#'   fit.model = "binomial",
#'   verbose = TRUE
#' )
#' 
#' # Check model components
#' str(null_model_pois)
#' str(null_model_bin)
#' 
#' # Access specific components
#' coefficients <- null_model_pois$coefficients
#' convergence <- null_model_pois$converged
#' computation_time <- null_model_pois$times
#' }
#'
#' @seealso
#' \code{\link{spark.fit}}, \code{\link{spark.ai}}, \code{\link{SpatialVG}},
#' \code{\link{glm}}
#'
#' @importFrom stats poisson as.formula glm binomial model.frame na.pass
#' 
#' @export
#' 
spark.null <- function(y, 
				covariates = NULL, 
				lib.size = NULL, 
				fit.model = "poisson",
				fit.maxiter = 100, 
				fit.tol = 1e-5, verbose = FALSE) {
	
	if(fit.model == "poisson"){
		if(is.null(lib.size)){stop("The parameter 'lib.size' is required for Poisson model!")}
		family <- poisson(link = "log")
		if(is.null(covariates)){
			fmla <- as.formula(paste0("y ~ 1 + offset(log(lib.size))"))
		}else{
			fmla <- as.formula(paste0("y ~ covariates + offset(log(lib.size))"))
		}## end fi
	}else if(fit.model == "binomial"){
		family <- binomial(link = "logit")
		if(is.null(covariates)){
			fmla <- as.formula(paste0("y ~ 1"))
		}else{
			fmla <- as.formula(paste0("y ~ covariates"))
		}## end fi
	}## end fi
	
	## run null model as initial
	model0 <- try(glm(formula = fmla, family = family))
	idx <- match(rownames(model.frame(formula = fmla, na.action = na.omit)),rownames(model.frame(formula = fmla, na.action = na.pass)))
	
	## model to fit
	t1 <- system.time(model1 <- try(spark.fit(model0, maxiter = fit.maxiter, tol = fit.tol, verbose = verbose)))
	
	if((class(model1) != "try-error")&&(!any(is.na(model1$Y)))){
		#cat(paste("SpatialVG::tau = ", model1$theta,"\n"))
		model1$times <- t1[3] # running time for each gene	
	}else{
		model1$converged <- FALSE
	}## end fi
	model1$idx <- idx
	## return the fitted model0
	return(model1)
}## end func


#' Fit spatial models for various distribution families
#'
#' This function fits spatial models for Gaussian, Poisson, and Binomial
#' distributions using appropriate estimation methods. For Gaussian data,
#' it uses direct calculation; for count and binary data, it employs the
#' Average Information algorithm with iterative refinement to handle
#' variance component estimation and boundary cases.
#'
#' @param model0 An initial generalized linear model object fitted by \code{glm}.
#'              This serves as the starting point for spatial model fitting.
#'              Must be of family "gaussian", "poisson", or "binomial".
#' @param maxiter Maximum number of iterations for the estimation algorithm.
#'               Default is 500.
#' @param tol Tolerance threshold for convergence. The algorithm stops when
#'           parameter changes fall below this value. Default is 1e-5.
#' @param verbose Logical indicating whether to print iteration progress and
#'               debugging information. Default is FALSE.
#'
#' @return Returns a list with the following components (specific components
#'         may vary by family):
#'   \itemize{
#'     \item \code{theta}: Estimated variance components
#'     \item \code{coefficients}: Estimated fixed effects coefficients
#'     \item \code{linear.predictors}: Linear predictors (η = Xβ + offset)
#'     \item \code{fitted.values}: Fitted values on the response scale
#'     \item \code{Y}: Working response vector
#'     \item \code{residuals}: Response residuals (y - μ)
#'     \item \code{Py}: Projected working response vector
#'     \item \code{X}: Design matrix for fixed effects
#'     \item \code{D}: Diagonal weights matrix for the working response
#'     \item \code{converged}: Logical indicating whether the algorithm converged
#'           (for Poisson and Binomial families)
#'   }
#'
#' @details
#' This function provides a unified interface for fitting spatial models
#' across different distribution families:
#'
#' \describe{
#'   \item{Gaussian}{Uses direct calculation of variance components based
#'         on residuals. Fast and exact for normally distributed data.}
#'   \item{Poisson}{Employs the Average Information algorithm with iterative
#'         refinement to handle overdispersion and spatial correlation in
#'         count data.}
#'   \item{Binomial}{Uses the same iterative approach as Poisson but for
#'         binary/binary data, handling spatial correlation in binary outcomes.}
#' }
#'
#' For Poisson and Binomial families, the function implements an iterative
#' refinement process that:
#' \enumerate{
#'   \item Fits the initial model using the Average Information algorithm
#'   \item Identifies variance components that are effectively zero
#'   \item Refits the model with these components fixed
#'   \item Repeats until stability is achieved or maximum iterations reached
#' }
#'
#' This approach ensures robust estimation even when some variance components
#' are at the boundary of the parameter space.
#'
#' @examples
#' \dontrun{
#' # Gaussian spatial model
#' set.seed(123)
#' n <- 100
#' x <- rnorm(n)
#' y_gaussian <- 1 + 0.5*x + rnorm(n, sd = 0.5)
#' model0_gauss <- glm(y_gaussian ~ x, family = gaussian)
#' result_gauss <- spark.fit(model0_gauss, verbose = TRUE)
#'
#' # Poisson spatial model for count data
#' mu_pois <- exp(1 + 0.3*x + rnorm(n, sd = 0.2))
#' y_pois <- rpois(n, mu_pois)
#' model0_pois <- glm(y_pois ~ x, family = poisson)
#' result_pois <- spark.fit(model0_pois, maxiter = 100, tol = 1e-4)
#'
#' # Binomial spatial model for binary data
#' prob_bin <- plogis(0.5 + 0.4*x + rnorm(n, sd = 0.3))
#' y_bin <- rbinom(n, 1, prob_bin)
#' model0_bin <- glm(y_bin ~ x, family = binomial)
#' result_bin <- spark.fit(model0_bin, verbose = TRUE)
#'
#' # Compare results across families
#' summary(result_gauss$coefficients)
#' summary(result_pois$coefficients)
#' summary(result_bin$coefficients)
#' }
#'
#' @references
#' Gilmour, A. R., Thompson, R., & Cullis, B. R. (1995). Average information
#' REML: an efficient algorithm for variance parameter estimation in linear
#' mixed models. Biometrics, 51(4), 1440-1450.
#'
#' Breslow, N. E., & Clayton, D. G. (1993). Approximate inference in
#' generalized linear mixed models. Journal of the American Statistical
#' Association, 88(421), 9-25.
#'
#' @seealso
#' \code{\link{spark.ai}}, \code{\link{glm}}, \code{\link{SpatialVG}}
#'
#'@importFrom stats model.matrix
#'
#' @export
spark.fit <- function(model0, 
                      maxiter = 500, 
                      tol = 1e-5, 
                      verbose = FALSE) {

	if(model0$family$family == "gaussian"){
		tau <- rep(0, 2)
		# fitting vector
		y <- model0$y
		num_cell <- length(y)
		offset <- model0$offset
		if(is.null(offset)) {offset <- rep(0, num_cell)}
	
		#family <- model0$family
		eta <- model0$linear.predictors
		mu  <- model0$fitted.values
		mu.eta <- model0$family$mu.eta(eta)
		D <- mu.eta/sqrt(model0$family$variance(mu))
		tau[2] <- sum(model0$residuals*model0$residuals)/(length(model0$residuals))
		# working vector
		Y <- eta - offset + (y - mu)/mu.eta	#eta is log(y),offset is log(N)
		X <- model.matrix(model0) ## 
		alpha <- model0$coef
		H <- tau[2]*rep(1,num_cell)
		Hinv <- 1.0/H
	
		## number of covariates, including the intercept
		num_cv <- ncol(X)
		
		if(num_cv > 1){## for multiple covariates, time-consuming
			Hinv <- diag(Hinv)
			HinvX <- crossprod(Hinv, X)
			XHinvX <- crossprod(X, HinvX)
			P <- try(Hinv - tcrossprod(tcrossprod(HinvX, chol2inv(chol( XHinvX ))), HinvX))
			Py <- crossprod(P, Y)
			rm(P)
		}else{## only intercept include the model, fit the model more effeciently
			## modified by sun, 2019-8-10 11:04:26
			HinvX <- Hinv
			XHinvX <- sum(HinvX)	
			Py <- Hinv*Y - HinvX%*%(t(HinvX)%*%Y)/XHinvX
		}# end fi num_cv (HinvX%*%t(HinvX)/XHinvX)%*%Y
		model1 <- list(theta = tau, coefficients = alpha, linear.predictors = eta, fitted.values = mu, Y = Y, residuals = y - mu, Py = Py, X = X, D = D)
	}else if(model0$family$family == "poisson"){
		## the number of variance component
		num_vc <- 1	
		fixtau.old <- rep(0, num_vc + 1)
		## to use average information method to fit alternative model
		model1 <- spark.ai(model0, num_vc, maxiter = maxiter, tol = tol, verbose = verbose)
		fixtau.new <- 1*(model1$theta < 1.01 * tol)
		count_num <- 1
		## while(any(fixtau.new != fixtau.old & count_num<50)) {
		while(any(fixtau.new != fixtau.old & count_num<maxiter)) {
			count_num <- count_num + 1
			fixtau.old <- fixtau.new
			## to use average information method to fit alternative model
			model1 <- spark.ai(model0, num_vc, fixtau = fixtau.old, maxiter = maxiter, tol = tol, verbose = verbose)
			fixtau.new <- 1*(model1$theta < 1.01 * tol)
		}# end while
	}else if(model0$family$family == "binomial"){
		## the number of variance component
		num_vc <- 1	
		fixtau.old <- rep(0, num_vc + 1)
		## to use average information method to fit alternative model
		model1 <- spark.ai(model0, num_vc, maxiter = maxiter, tol = tol, verbose = verbose)
		fixtau.new <- 1*(model1$theta < 1.01 * tol)
		count_num <- 1
		## while(any(fixtau.new != fixtau.old & count_num<50)) {
		while(any(fixtau.new != fixtau.old & count_num < maxiter)) {
			count_num <- count_num + 1
			fixtau.old <- fixtau.new
			## to use average information method to fit alternative model
			model1 <- spark.ai(model0, num_vc, fixtau = fixtau.old, maxiter = maxiter, tol = tol, verbose = verbose)
			fixtau.new <- 1*(model1$theta < 1.01 * tol)
		}# end while
	}# end fi
	
	# return the results
	return(model1)
}# end func


#' Fit spatial models using Average Information algorithm
#'
#' This function fits binary-based or count-based spatial models using the
#' Average Information (AI) algorithm, which is particularly efficient for
#' generalized linear mixed models with multiple variance components. It is
#' designed for Poisson and binomial spatial models and handles both fixed
#' and random effects.
#'
#' @param model0 An initial generalized linear model object fitted by \code{glm}.
#'              This serves as the starting point for the spatial model fitting.
#' @param num_vc Number of variance components (random effects) in the model.
#'              This includes both the dispersion parameter and spatial
#'              variance components.
#' @param tau Initial values for variance components parameters. Should be a
#'           vector of length \code{num_vc + 1} where the first element is
#'           typically the dispersion parameter. Default is \code{rep(0.1, num_vc + 1)}.
#' @param fixtau Indicator vector specifying which variance components should
#'              be fixed during optimization. Use 1 to fix, 0 to estimate.
#'              Default is \code{rep(0, num_vc + 1)} (all components estimated).
#' @param maxiter Maximum number of iterations for the AI algorithm. Default is 500.
#' @param tol Tolerance threshold for convergence. The algorithm stops when
#'           parameter changes fall below this value. Default is 1e-5.
#' @param verbose Logical indicating whether to print iteration progress and
#'               debugging information. Default is FALSE.
#'
#' @return Returns a list with the following components:
#'   \itemize{
#'     \item \code{theta}: Estimated variance components
#'     \item \code{coefficients}: Estimated fixed effects coefficients
#'     \item \code{linear.predictors}: Linear predictors (η = Xβ + offset)
#'     \item \code{fitted.values}: Fitted values on the response scale
#'     \item \code{Y}: Working response vector
#'     \item \code{residuals}: Response residuals (y - μ)
#'     \item \code{Py}: Projected working response vector
#'     \item \code{cov}: Covariance matrix of fixed effects
#'     \item \code{X}: Design matrix for fixed effects
#'     \item \code{D}: Diagonal weights matrix for the working response
#'     \item \code{converged}: Logical indicating whether the algorithm converged
#'   }
#'
#' @details
#' The Average Information algorithm is an efficient method for estimating
#' variance components in generalized linear mixed models. This implementation:
#'
#' \itemize{
#'   \item Supports both Poisson and binomial families for count and binary data
#'   \item Handles multiple variance components with optional fixing of parameters
#'   \item Uses efficient matrix computations for both single and multiple covariates
#'   \item Includes robust convergence checking and numerical stability measures
#'   \item Provides fallback to initial estimates when numerical issues occur
#' }
#'
#' The algorithm proceeds as follows:
#' \enumerate{
#'   \item Initialize parameters from the base GLM model
#'   \item Compute the working response vector and weights
#'   \item Estimate initial variance components using method of moments
#'   \item Iteratively update parameters using the Average Information matrix
#'   \item Check convergence based on relative parameter changes
#'   \item Handle numerical instability with fallback strategies
#' }
#'
#' For models with only an intercept term (single covariate), optimized
#' computational pathways are used for efficiency.
#'
#' @examples
#' \dontrun{
#' # Example with Poisson spatial model
#' library(stats)
#'
#' # Generate example data
#' set.seed(123)
#' n <- 100
#' x <- rnorm(n)
#' spatial_effect <- rnorm(n, sd = 0.5)
#' mu <- exp(1 + 0.5*x + spatial_effect)
#' y <- rpois(n, mu)
#'
#' # Fit initial Poisson GLM
#' model0 <- glm(y ~ x, family = poisson)
#'
#' # Fit spatial model with one variance component
#' result <- spark.ai(
#'   model0 = model0,
#'   num_vc = 1,
#'   tau = c(1, 0.1),  # dispersion = 1, spatial variance = 0.1
#'   fixtau = c(1, 0),  # fix dispersion, estimate spatial variance
#'   maxiter = 100,
#'   tol = 1e-4,
#'   verbose = TRUE
#' )
#'
#' # Check results
#' summary(result$coefficients)
#' result$theta
#' result$converged
#'
#' # Example with binomial spatial model
#' y_binary <- rbinom(n, 1, plogis(0.5*x + spatial_effect))
#' model0_bin <- glm(y_binary ~ x, family = binomial)
#'
#' result_bin <- spark.ai(
#'   model0 = model0_bin,
#'   num_vc = 1,
#'   maxiter = 100
#' )
#' }
#'
#' @references
#' Gilmour, A. R., Thompson, R., & Cullis, B. R. (1995). Average information
#' REML: an efficient algorithm for variance parameter estimation in linear
#' mixed models. Biometrics, 51(4), 1440-1450.
#'
#' Breslow, N. E., & Clayton, D. G. (1993). Approximate inference in
#' generalized linear mixed models. Journal of the American Statistical
#' Association, 88(421), 9-25.
#'
#' @seealso
#' \code{\link{glm}}, \code{\link{SpatialVG}}, \code{\link{spark.x}}
#'
#' @importFrom stats model.matrix median
#' 
#' @export
spark.ai <- function(model0, 
                     num_vc, 
                     tau = rep(0.1, num_vc+1), 
                     fixtau = rep(0, num_vc+1), 
                     maxiter = 500, 
                     tol = 1e-5, 
                     verbose = FALSE){

	## fitting vector
	y <- model0$y

	num_cell <- length(y)
	offset <- model0$offset
	if(is.null(offset)) {offset <- rep(0, num_cell)}
	
	family <- model0$family
	eta <- model0$linear.predictors
	mu  <- model0$fitted.values
	mu.eta <- family$mu.eta(eta)
	D <- mu.eta/sqrt(model0$family$variance(mu))

	## working vector
	Y <- eta - offset + (y - mu)/mu.eta	#eta is log(y),offset is log(N)
	X <- model.matrix(model0) ## 
	alpha <- model0$coef
	
	## number of covariates, including the intercept
	num_cv <- ncol(X)
	## fix first parameter, dispersion
	if(family$family %in% c("poisson", "binomial")){
		tau[1] <- 1
		fixtau[1] <- 1
	}## end fi

	## find the initial results for tau
	idxtau <- which(fixtau == 0)
	num_cov_mat2 <- sum(fixtau == 0) # is equal to 1
	if(num_cov_mat2 > 0){
		tau[fixtau == 0] <- rep(min(0.9, var(Y)/(num_vc + 1)), num_cov_mat2)
		H <- tau[1]*(1/D^2)
		H <- H + tau[2]
		Hinv <- 1.0/H
	  
		if(num_cv > 1){# for multiple covariates, time-consuming
			Hinv <- diag(Hinv)
			HinvX <- crossprod(Hinv, X)
			XHinvX <- crossprod(X, HinvX)
			P <- try(Hinv - tcrossprod(tcrossprod(HinvX, chol2inv(chol( XHinvX ))), HinvX))
			Py <- crossprod(P, Y)
			tau0 <- tau
			PAPY <- crossprod(P, Py)
			for(ik in 1:num_cov_mat2){
				##modified by sun, 2019-4-13 19:01:22
				tau[idxtau[ik]] <- max(0, tau0[idxtau[ik]] + tau0[idxtau[ik]]^2 * (crossprod(Y, PAPY) - sum(diag(P)))/num_cell)
			}# end for ik loop
			rm(P)
			gc()
		}else{## only intercept include the model, fit the model more efficiently
			## modified by sun, 2019-4-13 18:57:06
			HinvX <- Hinv
			XHinvX <- sum(HinvX)	
			Py <- Hinv*Y - Hinv*as.numeric(t(HinvX)%*%Y)/XHinvX
			diagP <- Hinv - HinvX*HinvX/XHinvX
  		
			tau0 <- tau
			## modified by sun, 2019-5-20 17:03:52
			#PAPY <- crossprod(P, Py)
			PAPY <- as.numeric( Hinv*Py - Hinv*as.numeric(t(HinvX)%*%Py)/XHinvX)
			for(ik in 1:num_cov_mat2){
				##modified by sun, 2019-4-13 19:01:22
				tau[idxtau[ik]] <- max(0, tau0[idxtau[ik]] + tau0[idxtau[ik]]^2 * (crossprod(Y, PAPY) - sum(diagP))/num_cell)
			}# end for ik loop
			rm(diagP)
			gc()
			Py <- t(Py)
		}# end fi num_cv
		rm(H)
		rm(Hinv)
		rm(HinvX)
		#rm(Py)
		rm(PAPY)
		gc()
	}## end if num_cov_mat2
	
	init_tau <- tau
	## with reasonable initial values
	tau[which(tau>10)] <- 0.5
	tau[which(tau<0)] <- 0.5
	##cat(paste0("initial tau=",tau,"\n"))
	for (iter in seq_len(maxiter)){	
		alpha0 <- alpha
		tau0 <- tau
		if(num_cv > 1){# Cppp code to speed up
		  model1 <- CovariatesAI(Y, X, D^2, tau, fixtau, tol)
		}else{# more faster version becasue X is one vector
		  model1 <- noCovariatesAI(Y, X, D^2, tau, fixtau, tol)
		}## end fi
	
		tau <- as.numeric(model1$tau)
		cov <- as.matrix(model1$cov)
		alpha <- as.numeric(model1$alpha)
		eta <- as.numeric(model1$eta) + offset
		
		mu <- family$linkinv(eta)
		mu.eta <- family$mu.eta(eta)
		D <- mu.eta/sqrt(family$variance(mu))	
		Y <- eta - offset + (y - mu)/mu.eta
		## @@@to avoid the strange values@@@, modified by sun, 2019-5-20 18:59:56
		Y[which(abs(Y)>1e3)] <- median(Y)#@@
		## @@@inverse normal to avoid the outlier@@@
		##@@@Y <- qnorm((rank(Y, na.last="keep", ties.method="random")-0.5)/sum(!is.na(Y)));
		# stop rule
		if(2*max(abs(alpha - alpha0)/(abs(alpha) + abs(alpha0) + tol), abs(tau - tau0)/(abs(tau) + abs(tau0) + tol)) < tol){	
			break;
		}## end fi
		if(max(tau) > tol^(-2)|any(is.infinite(D))|any(is.infinite(mu))|any(is.infinite(eta)) ){
			###== give the initial values obtained by linear regression ==###
			## modified by sun, 2019-5-20 19:34:10
			y <- model0$y
			family <- model0$family
			eta <- model0$linear.predictors
			mu  <- model0$fitted.values
			mu.eta <- family$mu.eta(eta)
			D <- mu.eta/sqrt(model0$family$variance(mu))
			model1$Py <- Py # obtained from initial value
			##working vector
			Y <- eta - offset + (y - mu)/mu.eta	#eta is log(y),offset is log(N)
			X <- model.matrix(model0) ##
			alpha <- model0$coef
			tau <- init_tau
			##
			iter <- maxiter
			break;
		}# end fi
	}## end for

	converged <- ifelse(iter < maxiter, TRUE, FALSE)	
	## return results
	return(list(theta = tau, coefficients = alpha, linear.predictors = eta, fitted.values = mu, Y = Y, residuals = y - mu, Py = model1$Py, cov = cov, X = X, D = D, converged = converged))
	#return(list(theta = tau, coefficients = alpha, Y = Y, D = D, converged = converged))
}## end function


#####################################################
#' SPARK-G: Fast Gaussian-based spatial pattern detection
#'
#' This function implements the SPARK-G method for detecting spatially variable
#' genes using a Gaussian modeling framework. It is designed for normalized
#' gene expression data and provides fast, scalable testing of spatial patterns
#' using kernel-based methods with Satterthwaite approximation for p-value computation.
#'
#' @param norm_counts Normalized gene expression matrix with p genes as rows and
#'                   n cells as columns. The data should be variance-stabilized
#'                   or otherwise normalized to approximate Gaussian distribution.
#' @param Kmat Kernel matrix of dimension n x n that encodes spatial relationships
#'            between cells. Common kernels include Gaussian, cosine, or linear kernels.
#' @param covariates Covariate matrix with n cells as rows and c covariates as
#'                 columns. Used to adjust for technical and biological confounding
#'                 factors. Must include an intercept term if desired.
#' @param Xdagger Precomputed pseudo-inverse of the covariate matrix. This can
#'               be computed using \code{pracma::pinv(covariates)}. Precomputing
#'               this matrix improves computational efficiency when testing
#'               multiple genes.
#' @param verbose Logical indicating whether to print progress messages and
#'               debugging information. Default is FALSE.
#'
#' @return Returns a data.frame with the following columns:
#'   \itemize{
#'     \item \code{pvalues}: P-values testing the significance of spatial patterns
#'     \item \code{stat_sw}: Test statistics measuring spatial pattern strength
#'   }
#'   Rows correspond to genes, with row names preserved from the input norm_counts matrix.
#'
#' @details
#' SPARK-G is a fast method for detecting spatially variable genes that:
#' \itemize{
#'   \item Assumes Gaussian-distributed normalized expression data
#'   \item Uses kernel methods to capture spatial covariance structures
#'   \item Adjusts for covariates using projection methods
#'   \item Employs Satterthwaite approximation for efficient p-value computation
#'   \item Scales linearly with the number of genes
#' }
#'
#' The method works by:
#' \enumerate{
#'   \item Projecting the kernel matrix to remove covariate effects
#'   \item Computing the expected value and variance of the test statistic under the null
#'   \item Calculating a scaled chi-square approximation for the test statistic
#'   \item Computing p-values using the Satterthwaite approximation
#' }
#'
#' The test statistic follows approximately a scaled chi-square distribution:
#' \deqn{S \sim k \cdot \chi^2_v}
#' where the scale parameter \eqn{k} and degrees of freedom \eqn{v} are estimated
#' from the kernel matrix.
#'
#' Key advantages:
#' \itemize{
#'   \item Computational efficiency for large numbers of genes
#'   \item Robust to various spatial correlation structures
#'   \item Proper adjustment for technical and biological covariates
#'   \item No need for permutation or resampling
#' }
#'
#' @examples
#' \dontrun{
#' # Load required libraries
#' library(pracma)
#'
#' # Generate example normalized expression data
#' n_genes <- 1000
#' n_cells <- 500
#' norm_counts <- matrix(rnorm(n_genes * n_cells), nrow = n_genes, ncol = n_cells)
#' rownames(norm_counts) <- paste0("Gene_", 1:n_genes)
#'
#' # Create spatial kernel matrix (Gaussian kernel)
#' spatial_coords <- cbind(x = runif(n_cells), y = runif(n_cells))
#' dist_matrix <- as.matrix(dist(spatial_coords))
#' Kmat <- exp(-dist_matrix^2 / (2 * 0.1^2))  # Gaussian kernel with bandwidth 0.1
#'
#' # Set up covariates (including intercept)
#' covariates <- cbind(intercept = 1, batch = sample(1:3, n_cells, replace = TRUE))
#' Xdagger <- pracma::pinv(covariates)  # Precompute pseudo-inverse
#'
#' # Run SPARK-G
#' results <- spark.g(
#'   norm_counts = norm_counts,
#'   Kmat = Kmat,
#'   covariates = covariates,
#'   Xdagger = Xdagger,
#'   verbose = TRUE
#' )
#'
#' # Identify significant spatially variable genes (FDR < 0.05)
#' sig_genes <- results[p.adjust(results$pvalues, method = "fdr") < 0.05, ]
#' head(sig_genes[order(sig_genes$pvalues), ])
#'
#' # Compare with other methods
#' plot(-log10(results$pvalues), main = "SPARK-G Results",
#'      xlab = "Gene Rank", ylab = "-log10 p-value")
#' abline(h = -log10(0.05), col = "red", lty = 2)
#' }
#'
#' @references
#' Sun, S., Zhu, J., & Zhou, X. (2020). Statistical analysis of spatial expression
#' patterns for spatially resolved transcriptomic studies. Nature Methods, 17(2), 193-200.
#'
#' Satterthwaite, F. E. (1946). An approximate distribution of estimates of variance
#' components. Biometrics Bulletin, 2(6), 110-114.
#'
#' @seealso
#' \code{\link{spark.x}}, \code{\link{SpatialVG}}, \code{\link[pracma]{pinv}},
#' \code{\link[stats]{pchisq}}
#'
#' @importFrom pbapply pbsapply
#' @importFrom stats var pchisq
#' @export
spark.g <- function(norm_counts, 
                    Kmat, 
                    covariates, 
                    Xdagger, 
                    verbose=FALSE){
  num_gene <- nrow(norm_counts)
  num_cell <- ncol(norm_counts)
  
  #covariates <- as.matrix(rep(1, num_cell))
  #Xdagger <- pinv(covariates)
  SK <- Kmat - covariates%*%(Xdagger%*%Kmat)
  
  ES <- 0.5*sum(diag(SK))
  VS <- 0.5*sum(t(SK)*SK) ## tr(SKSK)
  
  kk <- VS/(2*ES)
  vv <- 2*ES^2/VS
  
  res_sw <- pbapply::pbsapply(X = 1:num_gene, FUN = function(x) {
    yy <- norm_counts[x,]
    meanY <- mean(yy)
    varY <- var(yy)
    SY <- yy - meanY
    S0 <- 0.5*crossprod(SY, crossprod(Kmat, SY))/varY
    sw_pval <- pchisq(S0/kk, vv, lower.tail = FALSE)
    return(c(sw_pval, S0))
  })
  colnames(res_sw) <- rownames(norm_counts)
  rownames(res_sw) <- c("pvalues", "stat_sw")
  return(t(res_sw))
}## end func


################
#' SPARK-X: Non-parametric spatial pattern detection for transcriptomics
#'
#' This function implements the SPARK-X method for detecting spatially variable
#' genes in spatial transcriptomics data using a non-parametric framework.
#' It tests for spatial patterns in gene expression without assuming specific
#' distributional forms, making it robust and scalable for large datasets.
#'
#' @param counts Gene expression matrix with n cells as rows and p genes as columns.
#'              Can be a dense matrix, data.frame, or sparseMatrix. For large
#'              datasets, sparseMatrix format is recommended for memory efficiency.
#' @param infomat Spatial information matrix with n cells as rows and d spatial
#'               dimensions as columns. Typically contains spatial coordinates
#'               (x, y) and optionally additional spatial features.
#' @param X_mat Optional covariate matrix with n cells as rows and c covariates
#'            as columns. Used to adjust for technical and biological confounding
#'            factors such as batch effects, sequencing depth, or cell cycle stage.
#' @param mc_cores Number of CPU cores to use for parallel computation. Default is 1.
#'               For large datasets, increasing this value can significantly
#'               reduce computation time.
#' @param verbose Logical indicating whether to print progress messages and
#'               debugging information. Default is TRUE.
#'
#' @return Returns a data.frame with the following columns:
#'   \itemize{
#'     \item \code{stat}: Test statistic measuring the strength of spatial patterning
#'     \item \code{pval}: P-value assessing the significance of spatial pattern
#'   }
#'   Rows correspond to genes, with row names preserved from the input counts matrix.
#'
#' @details
#' SPARK-X is a non-parametric method for detecting spatially variable genes that:
#' \itemize{
#'   \item Does not assume specific distributional forms (e.g., Poisson, negative binomial)
#'   \item Handles both raw counts and normalized expression data
#'   \item Adjusts for technical and biological confounding factors
#'   \item Scales efficiently to large datasets with thousands of cells and genes
#'   \item Uses efficient linear algebra and parallel computation
#' }
#'
#' The method works by:
#' \enumerate{
#'   \item Projecting gene expression onto spatial coordinate basis functions
#'   \item Computing test statistics that measure correspondence between expression
#'         and spatial patterns
#'   \item Adjusting for covariates if provided
#'   \item Computing p-values using eigenvalue decomposition and Davies/Liu methods
#' }
#'
#' Key advantages over parametric methods:
#' \itemize{
#'   \item Robust to overdispersion and zero-inflation in count data
#'   \item No need for parameter tuning or distributional assumptions
#'   \item Fast computation even for large spatial datasets
#'   \item Maintains good statistical power across diverse data types
#' }
#'
#' @examples
#' \dontrun{
#' # Load required libraries
#' library(Matrix)
#' 
#' # Generate example spatial transcriptomics data
#' n_cells <- 500
#' n_genes <- 1000
#' n_dims <- 2
#' 
#' # Expression matrix (sparse for efficiency)
#' counts <- Matrix::Matrix(rpois(n_cells * n_genes, 5), 
#'                         nrow = n_cells, ncol = n_genes, sparse = TRUE)
#' rownames(counts) <- paste0("Cell_", 1:n_cells)
#' colnames(counts) <- paste0("Gene_", 1:n_genes)
#' 
#' # Spatial coordinates
#' infomat <- cbind(x = runif(n_cells), y = runif(n_cells))
#' 
#' # Run SPARK-X without covariates
#' results <- spark.x(
#'   counts = t(counts),  # Note: spark.x expects genes as rows
#'   infomat = infomat,
#'   mc_cores = 4,
#'   verbose = TRUE
#' )
#' 
#' # Run with covariates (e.g., batch effects, sequencing depth)
#' covariates <- cbind(
#'   batch = sample(1:3, n_cells, replace = TRUE),
#'   depth = rnorm(n_cells)
#' )
#' 
#' results_adj <- spark.x(
#'   counts = t(counts),
#'   infomat = infomat,
#'   X_mat = covariates,
#'   mc_cores = 4
#' )
#' 
#' # Identify significant spatially variable genes (FDR < 0.05)
#' sig_genes <- results[which(p.adjust(results$pval, method = "fdr") < 0.05), ]
#' head(sig_genes[order(sig_genes$pval), ])
#' 
#' # Compare with and without covariate adjustment
#' plot(-log10(results$pval), -log10(results_adj$pval),
#'      xlab = "-log10 p-value (unadjusted)",
#'      ylab = "-log10 p-value (adjusted)")
#' abline(0, 1, col = "red")
#' }
#'
#' @references
#' Zhu, J., Sun, S., & Zhou, X. (2021). SPARK-X: non-parametric modeling enables
#' scalable and robust detection of spatial expression patterns for large spatial
#' transcriptomic studies. Genome Biology, 22(1), 184.
#'
#' @seealso
#' \code{\link{sparkx.sksg}}, \code{\link{sparkx_pval}}, \code{\link{SpatialVG}},
#' \code{\link[CompQuadForm]{davies}}
#'
#' @importFrom Matrix Matrix
#' @importFrom parallel mclapply
#' @export
#' 
spark.x <- function(counts,
                    infomat,
                    X_mat = NULL,
                    mc_cores = 1,
                    verbose = TRUE){
  
  if(is(counts, "matrix")){
    counts <- as(counts, "sparseMatrix")
  }## end fi
  
  geneName <- rownames(counts)
  if(sum(is.na(geneName))>0){geneName[is.na(geneName)]<- "NAgene"}
  
  if(is.null(X_mat)){	
    Xinfomat <- apply(infomat, 2, scale,scale=FALSE)
    loc_inv <- solve(crossprod(Xinfomat, Xinfomat))
    kmat_first <- Xinfomat %*% loc_inv
    
    LocDim <- ncol(infomat)
    Klam <- eigen(crossprod(Xinfomat, kmat_first), only.values=TRUE)$values
    EHL <- counts%*%Xinfomat
    numCell <- nrow(Xinfomat)
    
    adjust_nominator <- as.vector(sp_sums_Rcpp(counts^2, TRUE))
    vec_stat <- apply(EHL,1,function(x){x%*%loc_inv%*%as.matrix(x)})*numCell/adjust_nominator
    
    vec_ybar <- as.vector(sp_means_Rcpp(counts,TRUE))
    vec_ylam <- unlist(parallel::mclapply(1:nrow(counts),function(x){1-numCell*vec_ybar[x]^2/adjust_nominator[x]},mc.cores=mc_cores))
    vec_daviesp <- unlist(parallel::mclapply(1:nrow(counts),function(x){sparkx_pval(x,vec_ylam,Klam,vec_stat)},mc.cores=mc_cores))
    res_sparkx <- as.data.frame(cbind(vec_stat,vec_daviesp))
  }else{
    ## check if it is fast or not:YES, much fast
    ## otherwise, too much memory required
    if(ncol(counts)<30000){
      counts <- as.matrix(counts)
    }## end fi
    
    numCell <- nrow(infomat)
    XTX_inv <- solve(crossprod(X_mat, X_mat))
    Xadjust_mat <- crossprod(infomat,X_mat)%*%crossprod(XTX_inv, t(X_mat))
    Xinfomat <- infomat - t(Xadjust_mat)
    
    # info_inv <- solve(crossprod(infomat,infomat))
    info_inv <- solve(crossprod(Xinfomat, Xinfomat))
    
    kmat_first <- Xinfomat %*% info_inv
    LocDim <- ncol(Xinfomat)
    
    Klam <- eigen(crossprod(Xinfomat, kmat_first), only.values=T)$values
    
    res_sparkx_list <- parallel::mclapply(X=1:nrow(counts),FUN=sparkx.sksg,
                                 expmat= counts,
                                 xmat = X_mat,
                                 scaleinfo = Xinfomat,
                                 numDim = LocDim,
                                 lambda_K = Klam,
                                 loc_inv = info_inv,
                                 mc.cores=mc_cores,
                                 verbose=verbose)
    res_sparkx <- as.data.frame(do.call(rbind,res_sparkx_list))
  }
  colnames(res_sparkx) <- c("stat","pval")
  rownames(res_sparkx) <- geneName
  
  return(res_sparkx)
}## end func



#' Single gene testing with non-parametric spatial framework (SPARK-X)
#'
#' This function performs spatial pattern testing for a single gene using
#' the SPARK-X non-parametric framework. It computes test statistics and
#' p-values by comparing gene expression patterns to spatial covariance
#' structures while adjusting for potential confounding factors.
#'
#' @param igene Index of the gene to test within the expression matrix.
#' @param expmat Gene expression matrix (sparseMatrix format) with n cells
#'              as rows and p genes as columns.
#' @param xmat Covariate matrix with n cells as rows and c covariates as
#'            columns. Used to adjust for technical and biological confounding
#'            factors.
#' @param scaleinfo Scaled and centered spatial coordinates matrix with n cells
#'                as rows and d spatial dimensions as columns.
#' @param numDim Number of spatial dimensions (typically 2 for 2D spatial data).
#' @param lambda_K Eigenvalues of the spatial kernel being tested, representing
#'                the spatial covariance structure.
#' @param loc_inv Inverse of the inner product matrix of spatial coordinates,
#'              used in the spatial test statistic computation.
#' @param verbose Logical indicating whether to print progress messages.
#'              Useful for debugging and monitoring large analyses.
#'
#' @return Returns a numeric vector of length 2 containing:
#'   \itemize{
#'     \item Test statistic measuring the strength of spatial patterning
#'     \item P-value assessing the significance of the spatial pattern
#'   }
#'
#' @details
#' This function implements the core statistical test for the SPARK-X method,
#' which detects spatially variable genes without assuming specific distributional
#' forms for the expression data. Key aspects:
#'
#' \itemize{
#'   \item \strong{Non-parametric approach}: Does not assume Poisson or negative
#'         binomial distributions, making it robust to various data types
#'   \item \strong{Covariate adjustment}: Removes effects of known confounding
#'         factors before spatial testing
#'   \item \strong{Eigenvalue decomposition}: Uses kernel eigenvalues to capture
#'         spatial covariance structure
#'   \item \strong{Davies/Liu approximation}: Computes exact p-values using
#'         quadratic form distributions with fallback to approximation when needed
#' }
#'
#' The test statistic is computed as:
#' \deqn{S = \frac{1}{\sigma^2} \cdot Y^T H L^{-1} H^T Y \cdot n}
#' where:
#' \itemize{
#'   \item \eqn{Y} is the gene expression vector
#'   \item \eqn{H} is the projection matrix for covariates
#'   \item \eqn{L} is the spatial coordinate matrix
#'   \item \eqn{n} is the number of cells
#'   \item \eqn{\sigma^2} is the variance estimate
#' }
#'
#' The p-value is computed using the Davies method for quadratic forms of
#' normal variables, with fallback to Liu approximation when necessary.
#'
#' @examples
#' \dontrun{
#' # Load required libraries
#' library(Matrix)
#' library(CompQuadForm)
#'
#' # Generate example data
#' n_cells <- 100
#' n_genes <- 50
#' n_covariates <- 3
#' n_dims <- 2
#'
#' # Expression matrix (sparse)
#' expmat <- Matrix::Matrix(rpois(n_cells * n_genes, 5), 
#'                         nrow = n_cells, ncol = n_genes, sparse = TRUE)
#'
#' # Covariate matrix
#' xmat <- matrix(rnorm(n_cells * n_covariates), nrow = n_cells)
#'
#' # Spatial coordinates (centered and scaled)
#' scaleinfo <- scale(matrix(runif(n_cells * n_dims), ncol = n_dims))
#'
#' # Kernel eigenvalues
#' lambda_K <- runif(n_dims, 0.5, 2.0)
#'
#' # Inverse of location inner product
#' loc_inv <- solve(crossprod(scaleinfo))
#'
#' # Test a single gene
#' result <- sparkx.sksg(
#'   igene = 1,
#'   expmat = expmat,
#'   xmat = xmat,
#'   scaleinfo = scaleinfo,
#'   numDim = n_dims,
#'   lambda_K = lambda_K,
#'   loc_inv = loc_inv,
#'   verbose = TRUE
#' )
#'
#' # Test multiple genes in parallel
#' all_results <- parallel::mclapply(
#'   1:10,
#'   sparkx.sksg,
#'   expmat = expmat,
#'   xmat = xmat,
#'   scaleinfo = scaleinfo,
#'   numDim = n_dims,
#'   lambda_K = lambda_K,
#'   loc_inv = loc_inv,
#'   verbose = FALSE,
#'   mc.cores = 4
#' )
#'
#' # Extract p-values
#' pvals <- sapply(all_results, function(x) x[2])
#' test_stats <- sapply(all_results, function(x) x[1])
#' }
#'
#' @references
#' Zhu, J., Sun, S., & Zhou, X. (2021). SPARK-X: non-parametric modeling enables
#' scalable and robust detection of spatial expression patterns for large spatial
#' transcriptomic studies. Genome Biology, 22(1), 184.
#'
#' Davies, R. B. (1980). The Distribution of a Linear Combination of
#' χ2 Random Variables. Journal of the Royal Statistical Society.
#' Series C (Applied Statistics), 29(3), 323-333.
#'
#' @seealso
#' \code{\link{sparkx_pval}}, \code{\link{SpatialVG}},
#' \code{\link[CompQuadForm]{davies}}, \code{\link[CompQuadForm]{liu}}
#'
#' @importFrom CompQuadForm davies liu
#' @export
#' 
sparkx.sksg <- function(igene, expmat, xmat, scaleinfo, numDim, lambda_K, loc_inv, verbose=TRUE){
  if(verbose){cat("gene",igene,"\n")}
  single_gene <- expmat[igene,]
  numCell <- length(single_gene)
  XTX_inv <- solve(crossprod(xmat,xmat))
  GTX <- crossprod(single_gene,xmat)
  Gadjust_mat <- GTX%*%tcrossprod(XTX_inv,GTX)
  # adj_nominator <- 1/as.vector(crossprod(single_gene,single_gene)-Gadjust_mat)
  adj_nominator <- 1/as.vector(crossprod(single_gene, single_gene))
  lambda_G <- as.vector(crossprod(single_gene,single_gene)-Gadjust_mat)*adj_nominator

  YHL <- single_gene%*%scaleinfo
  scoredavies <- adj_nominator*(YHL%*%loc_inv%*%t(YHL))*numCell
  
  Zsort <- sort(lambda_G*lambda_K,decreasing=TRUE)
  results_score <- try(CompQuadForm::davies(scoredavies, Zsort))
  if(class(results_score)!="try-error"){
    pout <- results_score$Qq
    if(pout<=0){
      pout <- CompQuadForm::liu(scoredavies, Zsort)
    }
  }else{
    pout <- NA
  }
  
  return(c(scoredavies,pout))
}## end func


#' Calculate SPARK-X p-values using Davies method
#'
#' This function computes p-values for the SPARK-X method using the Davies
#' algorithm for quadratic forms of normal variables. It is used internally
#' in spatial pattern detection to assess the significance of spatial
#' expression patterns for individual genes.
#'
#' @param igene Index of the gene for which to compute the p-value. This
#'             index corresponds to the position in the lambda_G and allstat
#'             vectors.
#' @param lambda_G A p-length numeric vector of eigenvalues for all genes.
#'                These eigenvalues are typically derived from the gene
#'                expression covariance structure.
#' @param lambda_K A d-length numeric vector of eigenvalues for the spatial
#'                kernel being tested. These capture the spatial covariance
#'                structure.
#' @param allstat A p-length numeric vector of test statistics for all genes.
#'               These statistics measure the strength of spatial patterning
#'               for each gene.
#'
#' @return Returns a single numeric value representing the p-value for the
#'         specified gene. If the Davies method fails or returns a non-positive
#'         p-value, the Liu approximation is used instead. Returns NA if
#'         both methods fail.
#'
#' @details
#' The SPARK-X method tests for spatially variable gene expression by
#' examining the correspondence between gene expression patterns and
#' spatial covariance structures. This function implements the statistical
#' test using the Davies algorithm, which computes the exact distribution
#' of quadratic forms of normal variables.
#'
#' The test statistic follows a mixture of chi-square distributions under
#' the null hypothesis of no spatial pattern. The eigenvalues (lambda_G
#' and lambda_K) define this mixture distribution, and the Davies algorithm
#' computes the probability of observing a test statistic as extreme as
#' the observed value.
#'
#' Key features:
#' \itemize{
#'   \item Uses Davies algorithm for exact p-value computation
#'   \item Falls back to Liu approximation when Davies fails
#'   \item Handles numerical edge cases gracefully
#'   \item Efficiently computes p-values gene by gene
#' }
#'
#' The function is typically used in parallel across multiple genes to
#' enable scalable spatial pattern detection in large datasets.
#'
#' @examples
#' \dontrun{
#' # Generate example eigenvalues and test statistics
#' n_genes <- 100
#' n_kernels <- 50
#' 
#' lambda_G <- runif(n_genes, 0.1, 2.0)  # Gene eigenvalues
#' lambda_K <- runif(n_kernels, 0.5, 3.0) # Kernel eigenvalues
#' allstat <- rchisq(n_genes, df = 2)     # Test statistics
#' 
#' # Compute p-value for a specific gene
#' gene_index <- 5
#' pval <- sparkx_pval(
#'   igene = gene_index,
#'   lambda_G = lambda_G,
#'   lambda_K = lambda_K,
#'   allstat = allstat
#' )
#' 
#' # Compute p-values for all genes in parallel
#' all_pvals <- parallel::mclapply(
#'   1:n_genes,
#'   sparkx_pval,
#'   lambda_G = lambda_G,
#'   lambda_K = lambda_K,
#'   allstat = allstat,
#'   mc.cores = 4
#' )
#' 
#' # Identify significant genes
#' significant_genes <- which(unlist(all_pvals) < 0.05)
#' }
#'
#' @references
#' Davies, R. B. (1980). The Distribution of a Linear Combination of
#' χ2 Random Variables. Journal of the Royal Statistical Society.
#' Series C (Applied Statistics), 29(3), 323-333.
#'
#' Liu, H., et al. (2009). Accurate and efficient p-value calculation
#' for the generalized chi-square distribution. Journal of
#' Computational Biology, 16(10), 1389-1398.
#'
#' @seealso
#' \code{\link[CompQuadForm]{davies}}, \code{\link[CompQuadForm]{liu}},
#' \code{\link{SpatialVG}}
#'
#' @importFrom CompQuadForm davies liu
#' @export
sparkx_pval <- function(igene,
                        lambda_G,
                        lambda_K,
                        allstat){
  Zsort <- sort(lambda_G[igene]*lambda_K,decreasing=TRUE)
  results_score <- try(CompQuadForm::davies(allstat[igene], Zsort))
  if(class(results_score)!="try-error"){
    pout <- results_score$Qq
    if(pout<=0){
      pout <- CompQuadForm::liu(allstat[igene], Zsort)
    }
  }else{
    pout <- NA
  }
  return(pout)
}## end func


#' Transform spatial coordinates using kernel-based transformations
#'
#' This function applies kernel-based transformations to spatial coordinates
#' to create enhanced feature representations for spatial pattern detection.
#' Different transformation functions can capture various aspects of spatial
#' relationships, such as periodicity, smoothness, or local connectivity.
#'
#' @param coord A numeric vector of spatial coordinates (x or y coordinates)
#'              for n spatial locations.
#' @param lker A smoothing or periodic parameter that controls the scale of
#'            the transformation. For Gaussian kernels, this is the bandwidth;
#'            for periodic kernels, this controls the period length.
#' @param transfunc The type of coordinate transformation function to apply.
#'                 Options include:
#'   \itemize{
#'     \item "gaussian": Gaussian kernel transformation
#'     \item "cosine": Cosine transformation for periodic patterns
#'     \item "linear": Linear transformation
#'     \item "matern": Matern covariance transformation
#'     \item "spline": Spline-based transformation
#'   }
#'   Default is "gaussian".
#'
#' @return Returns a transformed coordinate matrix with the same number of
#'         rows as the input coordinates, but with additional dimensions
#'         representing the kernel-transformed features. The exact dimensions
#'         depend on the transformation function used.
#'
#' @details
#' Coordinate transformations are essential for capturing complex spatial
#' patterns in kernel-based methods. This function provides several
#' transformation approaches:
#'
#' \describe{
#'   \item{Gaussian transformation}{Applies a Gaussian kernel to create
#'         smooth spatial features that capture local neighborhood relationships.
#'         Useful for detecting smooth spatial gradients.}
#'   \item{Cosine transformation}{Uses trigonometric functions to capture
#'         periodic spatial patterns, such as repeating tissue structures
#'         or wave-like expression patterns.}
#'   \item{Linear transformation}{Simple linear transformation that captures
#'         directional trends in spatial data.}
#'   \item{Matern transformation}{Applies Matern covariance functions that
#'         can capture spatial patterns with varying degrees of smoothness.}
#'   \item{Spline transformation}{Uses spline basis functions to capture
#'         complex non-linear spatial patterns.}
#' }
#'
#' The transformation enhances the original coordinates by creating new
#' features that better represent spatial relationships, which can improve
#' the performance of spatial pattern detection algorithms.
#'
#' @examples
#' \dontrun{
#' # Generate sample spatial coordinates
#' x_coords <- runif(100, 0, 10)
#' y_coords <- runif(100, 0, 10)
#'
#' # Apply Gaussian transformation
#' gaussian_features <- transloc_func(
#'   coord = x_coords,
#'   lker = 2.0,
#'   transfunc = "gaussian"
#' )
#'
#' # Apply cosine transformation for periodic patterns
#' cosine_features <- transloc_func(
#'   coord = y_coords,
#'   lker = 5.0,
#'   transfunc = "cosine"
#' )
#'
#' # Apply multiple transformations for comprehensive spatial representation
#' coords_matrix <- cbind(x_coords, y_coords)
#' transformed_coords <- apply(coords_matrix, 2, transloc_func,
#'                            lker = 3.0, transfunc = "gaussian")
#'
#' # Use in spatial pattern detection
#' spatial_features <- cbind(
#'   transloc_func(x_coords, lker = 2.0, transfunc = "gaussian"),
#'   transloc_func(y_coords, lker = 2.0, transfunc = "gaussian")
#' )
#' }
#'
#' @seealso
#' \code{\link{ComputeSingleKernelMat}}, \code{\link{SpatialVG}}
#'
#' @export
transloc_func <- function(coord, 
                          lker, 
                          transfunc = "gaussian"){
  ## for the simulation 
  coord <- scale(coord, scale=FALSE)
  
  #l <- quantile(abs(coord),probs=seq(0.2,1,by=0.2))
  if(transfunc == "gaussian"){
    out <- exp(-coord^2/(2*lker^2))
  }else if(transfunc == "cosine"){
    out <- cos(2*pi*coord/lker)
  }else if(transfunc == "linear"){
    out <- coord
  }## end fi
  return(out)
}## end func


#########################################
#             CODE END                  #
#########################################
