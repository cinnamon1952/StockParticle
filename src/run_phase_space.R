library(nonlinearTseries)
library(quantmod)
library(plot3D)

tickers <- c("AAPL", "MSFT", "GOOGL", "TSLA", "AMZN")
start_date <- "2020-01-01"
end_date <- "2025-01-01"

out_dir <- file.path(getwd(), "outputs_r")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

safe_timeLag <- function(x, technique, lag.max = 100) {
  tryCatch(
    timeLag(x, technique = technique, lag.max = lag.max, do.plot = FALSE),
    error = function(e) NA_integer_
  )
}

safe_estimateEmbeddingDim <- function(x, tau, max.embedding.dim = 15) {
  tryCatch(
    estimateEmbeddingDim(x, time.lag = tau, max.embedding.dim = max.embedding.dim, do.plot = FALSE),
    error = function(e) NA_integer_
  )
}

process_ticker <- function(ticker) {
  cat("\n", rep("=", 60), "\n")
  cat("Processing", ticker, "\n")
  cat(rep("=", 60), "\n")

  data <- getSymbols(ticker, src = "yahoo",
                     from = start_date, to = end_date,
                     auto.assign = FALSE)
  prices <- as.numeric(Cl(data))
  prices <- prices[!is.na(prices)]

  if (length(prices) < 100) {
    cat("Skipping", ticker, ": insufficient data\n")
    return(invisible(NULL))
  }

  log_prices <- log(prices)
  returns <- diff(log_prices)

  cat("Data points:", length(prices), "prices,", length(returns), "returns\n")

  cat("\n--- Time Lag Estimation ---\n")
  tau_p_acf <- safe_timeLag(prices, technique = "acf", lag.max = 100)
  tau_p_ami <- safe_timeLag(prices, technique = "ami", lag.max = 100)
  tau_r_acf <- safe_timeLag(returns, technique = "acf", lag.max = 100)
  tau_r_ami <- safe_timeLag(returns, technique = "ami", lag.max = 100)

  cat("Prices  - ACF:", tau_p_acf, ", AMI:", tau_p_ami, "\n")
  cat("Returns - ACF:", tau_r_acf, ", AMI:", tau_r_ami, "\n")

  # Prefer AMI; fall back to 1 if not available
  tau_p <- if (!is.na(tau_p_ami) && tau_p_ami >= 1) tau_p_ami else 1
  tau_r <- if (!is.na(tau_r_ami) && tau_r_ami >= 1) tau_r_ami else 1

  cat("\n--- Embedding Dimension Estimation ---\n")
  emb_dim_p <- safe_estimateEmbeddingDim(prices, tau = tau_p, max.embedding.dim = 15)
  emb_dim_r <- safe_estimateEmbeddingDim(returns, tau = tau_r, max.embedding.dim = 15)

  cat("Prices  - Optimal embedding dimension:", emb_dim_p, "\n")
  cat("Returns - Optimal embedding dimension:", emb_dim_r, "\n")

  m_p <- if (!is.na(emb_dim_p)) min(emb_dim_p, 3) else 3
  m_r <- if (!is.na(emb_dim_r)) min(emb_dim_r, 3) else 3

  cat("\n--- Building Takens Vectors ---\n")
  takens_p <- buildTakens(prices, embedding.dim = m_p, time.lag = tau_p)
  takens_r <- buildTakens(returns, embedding.dim = m_r, time.lag = tau_r)

  cat("Price Takens matrix shape:", dim(takens_p), "\n")
  cat("Return Takens matrix shape:", dim(takens_r), "\n")

  # Open PNG only once we're ready to plot
  out_file <- file.path(out_dir, paste0(ticker, "_takens.png"))
  png(filename = out_file, width = 1800, height = 1100, res = 150)
  on.exit(dev.off(), add = TRUE)

  # Correlation dimension is often finicky; keep it best-effort.
  cat("\n--- Correlation Dimension Estimation ---\n")
  cd_p <- tryCatch(
    corrDim(prices,
            min.embedding.dim = m_p,
            max.embedding.dim = m_p + 2,
            time.lag = tau_p,
            min.radius = 0.001,
            max.radius = 50,
            n.points.radius = 40,
            do.plot = FALSE),
    error = function(e) NULL
  )

  cd_r <- tryCatch(
    corrDim(returns,
            min.embedding.dim = m_r,
            max.embedding.dim = m_r + 2,
            time.lag = tau_r,
            min.radius = 0.001,
            max.radius = 1,
            n.points.radius = 40,
            do.plot = FALSE),
    error = function(e) NULL
  )

  cd_est_p <- tryCatch(
    if (!is.null(cd_p)) estimate(cd_p, regression.range = c(0.5, 2), use.embeddings = m_p) else NA,
    error = function(e) NA
  )
  cd_est_r <- tryCatch(
    if (!is.null(cd_r)) estimate(cd_r, regression.range = c(0.01, 0.1), use.embeddings = m_r) else NA,
    error = function(e) NA
  )

  cat("Price correlation dimension:", cd_est_p, "\n")
  cat("Return correlation dimension:", cd_est_r, "\n")

  par(mfrow = c(2, 3))

  scatter3D(takens_p[, 1], takens_p[, 2], takens_p[, 3],
            colvar = 1:nrow(takens_p),
            col = ramp.col(c("blue", "green", "yellow", "red")),
            pch = 19, cex = 0.3,
            main = paste(ticker, "- Price Phase Space\n(τ=", tau_p, ", m=", m_p, ")"),
            xlab = "P(t)", ylab = paste0("P(t+", tau_p, ")"),
            zlab = paste0("P(t+", 2*tau_p, ")"))

  scatter3D(takens_r[, 1], takens_r[, 2], takens_r[, 3],
            colvar = 1:nrow(takens_r),
            col = ramp.col(c("purple", "pink", "orange", "red")),
            pch = 19, cex = 0.3,
            main = paste(ticker, "- Return Phase Space\n(τ=", tau_r, ", m=", m_r, ")"),
            xlab = "r(t)", ylab = paste0("r(t+", tau_r, ")"),
            zlab = paste0("r(t+", 2*tau_r, ")"))

  plot(prices, type = "l", main = paste(ticker, "Price Series"),
       xlab = "Time", ylab = "Price")

  plot(returns, type = "l", main = paste(ticker, "Return Series"),
       xlab = "Time", ylab = "Log Return")

  if (!is.null(cd_p)) plot(cd_p, main = "Price Correlation Dimension") else plot.new()
  if (!is.null(cd_r)) plot(cd_r, main = "Return Correlation Dimension") else plot.new()

  par(mfrow = c(1, 1))

  cat("Saved plots to:", out_file, "\n")
  invisible(NULL)
}

for (ticker in tickers) {
  tryCatch(
    process_ticker(ticker),
    error = function(e) cat("Error processing", ticker, ":", conditionMessage(e), "\n")
  )
}

cat("\n", rep("=", 60), "\n")
cat("Processing complete!\n")
cat(rep("=", 60), "\n")