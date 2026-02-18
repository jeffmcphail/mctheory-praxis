# =====================================================================
#  McTheory Praxis — Makefile
# =====================================================================
#  Usage:
#    make help            Show available commands
#    make build           Build Docker image
#    make push            Push image to GitHub Container Registry
#    make pull            Pull image from registry (no source needed)
#    make surface-p1      Compute Phase 1 surfaces
#    make surface-p2      Compute Phase 2 surfaces
#    make test-docker     Run tests in container
#    make k8s-local       Deploy to local Kubernetes
# =====================================================================

.PHONY: help build test surface-p1 surface-p2 surface-p3 surface-p4 \
        surface-all surface-smoke shell test-docker merge \
        push pull login tag \
        k8s-local k8s-status k8s-logs k8s-delete clean status

IMAGE_NAME ?= praxis
IMAGE_TAG ?= latest
K8S_NS ?= praxis
DATA_DIR ?= $(CURDIR)/data

# ── GitHub Container Registry ─────────────────────────────────
# Set GHCR_USER to your GitHub username (or override on command line)
#   make push GHCR_USER=yourgithubname
#
# One-time setup on each machine:
#   1. Create a GitHub Personal Access Token (PAT) with packages:write scope
#      https://github.com/settings/tokens → "Generate new token (classic)"
#      Select scopes: write:packages, read:packages, delete:packages
#   2. Login: make login GHCR_USER=yourgithubname GHCR_TOKEN=ghp_xxxxx
#      (or: echo ghp_xxxxx | docker login ghcr.io -u yourgithubname --password-stdin)
#   3. After login, credentials are cached — no need to login again
#
GHCR_USER ?= mctheory
GHCR_REGISTRY = ghcr.io/$(GHCR_USER)
GHCR_IMAGE = $(GHCR_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

# ── Help ──────────────────────────────────────────────────────
help:
	@echo "McTheory Praxis — Available Commands"
	@echo "====================================="
	@echo ""
	@echo "Docker:"
	@echo "  make build            Build Docker image"
	@echo "  make test-docker      Run tests in container"
	@echo "  make shell            Interactive Python shell"
	@echo "  make status           Show container status"
	@echo ""
	@echo "Registry (ghcr.io):"
	@echo "  make login            One-time login to GitHub Container Registry"
	@echo "  make push             Build + push image to ghcr.io"
	@echo "  make pull             Pull image from ghcr.io (no source needed)"
	@echo "  make tag              Tag local image for registry"
	@echo ""
	@echo "Surface Compute:"
	@echo "  make surface-smoke    Quick validation (~2 min)"
	@echo "  make surface-p1       Phase 1: Core range (~1h on 8 cores)"
	@echo "  make surface-p2       Phase 2: Extended obs"
	@echo "  make surface-p3       Phase 3: Large universes"
	@echo "  make surface-p4       Phase 4: Full coverage"
	@echo "  make surface-all      All phases"
	@echo ""
	@echo "Surface Management:"
	@echo "  make merge SRC=<path> Merge remote surface DB into local"
	@echo "  make surface-status   Check surface build progress"
	@echo ""
	@echo "Kubernetes:"
	@echo "  make k8s-local        Deploy to local K8s"
	@echo "  make k8s-status       Show pod status"
	@echo "  make k8s-logs         Follow pod logs"
	@echo "  make k8s-delete       Remove K8s deployment"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run tests locally"
	@echo "  make test-docker      Run tests in Docker"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove containers and images"

# ── Docker ────────────────────────────────────────────────────
build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .
	@echo ""
	@echo "Built $(IMAGE_NAME):$(IMAGE_TAG)"

# ── Testing ───────────────────────────────────────────────────
test:
	python -m pytest tests/ -v --tb=short

test-docker: build
	docker run --rm $(IMAGE_NAME):$(IMAGE_TAG) test -v --tb=short

# ── Surface Compute ──────────────────────────────────────────
# These targets check for a local image. Use 'make build' or 'make pull' first.
_check-image:
	@docker image inspect $(IMAGE_NAME):$(IMAGE_TAG) >NUL 2>NUL || \
		(echo "ERROR: Image $(IMAGE_NAME):$(IMAGE_TAG) not found." && \
		 echo "  Run 'make build' (from source) or 'make pull' (from registry) first." && \
		 exit 1)

surface-smoke: _check-image
	docker run --rm -v "$(DATA_DIR):/app/data" \
		$(IMAGE_NAME):$(IMAGE_TAG) surface-build --smoke-test

surface-p1: _check-image
	docker run --rm -v "$(DATA_DIR):/app/data" \
		$(IMAGE_NAME):$(IMAGE_TAG) surface-build --phase 1

surface-p2: _check-image
	docker run --rm -v "$(DATA_DIR):/app/data" \
		$(IMAGE_NAME):$(IMAGE_TAG) surface-build --phase 2

surface-p3: _check-image
	docker run --rm -v "$(DATA_DIR):/app/data" \
		$(IMAGE_NAME):$(IMAGE_TAG) surface-build --phase 3

surface-p4: _check-image
	docker run --rm -v "$(DATA_DIR):/app/data" \
		$(IMAGE_NAME):$(IMAGE_TAG) surface-build --phase 4

surface-all: _check-image
	docker run --rm -v "$(DATA_DIR):/app/data" \
		$(IMAGE_NAME):$(IMAGE_TAG) surface-build --phase all

surface-status: _check-image
	docker run --rm -v "$(DATA_DIR):/app/data" \
		$(IMAGE_NAME):$(IMAGE_TAG) surface-build --status-only --phase all

# ── Surface Merge ────────────────────────────────────────────
merge: _check-image
ifndef SRC
	$(error SRC is required. Usage: make merge SRC=/path/to/remote/surfaces.duckdb)
endif
	docker run --rm \
		-v "$(DATA_DIR):/app/data" \
		-v "$(SRC):/remote/surfaces.duckdb:ro" \
		$(IMAGE_NAME):$(IMAGE_TAG) merge /remote/surfaces.duckdb

# ── Shell ────────────────────────────────────────────────────
shell: _check-image
	docker run --rm -it \
		-v "$(DATA_DIR):/app/data" \
		$(IMAGE_NAME):$(IMAGE_TAG) shell

# ── Registry (ghcr.io) ──────────────────────────────────────
# One-time per machine: make login GHCR_USER=you GHCR_TOKEN=ghp_xxx
login:
ifndef GHCR_TOKEN
	$(error GHCR_TOKEN required. Usage: make login GHCR_USER=yourgithub GHCR_TOKEN=ghp_xxxxx)
endif
	@echo "$(GHCR_TOKEN)" | docker login ghcr.io -u $(GHCR_USER) --password-stdin
	@echo ""
	@echo "Logged in to ghcr.io as $(GHCR_USER)"
	@echo "Credentials cached — no need to login again on this machine."

# Tag local image for registry
tag:
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(GHCR_IMAGE)
	@echo "Tagged: $(GHCR_IMAGE)"

# Build and push to registry (run on desktop / build machine)
push: build tag
	docker push $(GHCR_IMAGE)
	@echo ""
	@echo "Pushed: $(GHCR_IMAGE)"
	@echo "On any machine with docker login: make pull"

# Pull from registry (run on laptop / server — no source code needed)
pull:
	docker pull $(GHCR_IMAGE)
	docker tag $(GHCR_IMAGE) $(IMAGE_NAME):$(IMAGE_TAG)
	@echo ""
	@echo "Pulled and tagged locally as $(IMAGE_NAME):$(IMAGE_TAG)"
	@echo "Ready to run: make surface-p2"

# ── Kubernetes ───────────────────────────────────────────────
k8s-local: build
	kubectl create namespace $(K8S_NS) --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -k k8s/overlays/local
	@echo ""
	@echo "Deployed to namespace: $(K8S_NS)"
	@echo "  kubectl -n $(K8S_NS) get pods"

k8s-status:
	kubectl -n $(K8S_NS) get pods,jobs
	@echo ""
	kubectl -n $(K8S_NS) get pvc

k8s-logs:
	kubectl -n $(K8S_NS) logs -f -l app=praxis-surface --tail=100

k8s-delete:
	kubectl delete -k k8s/overlays/local --ignore-not-found
	@echo "Deleted K8s resources in $(K8S_NS)"

# ── Status ───────────────────────────────────────────────────
status:
	@docker images $(IMAGE_NAME) 2>/dev/null || echo "No image built"
	@echo ""
	@docker ps --filter "ancestor=$(IMAGE_NAME):$(IMAGE_TAG)" 2>/dev/null || true
	@echo ""
	@if [ -f "$(DATA_DIR)/surfaces.duckdb" ]; then \
		echo "Surface DB: $$(du -h $(DATA_DIR)/surfaces.duckdb | cut -f1)"; \
	else \
		echo "Surface DB: not found"; \
	fi

# ── Cleanup ──────────────────────────────────────────────────
clean:
	docker compose down 2>/dev/null || true
	docker rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true
	@echo "Cleaned up"
