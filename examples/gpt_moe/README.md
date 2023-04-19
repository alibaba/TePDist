:::info
Using model config in 'pretrain_gpt.json' by passing the '--config' argument
would run GPT model, while 'pretrain_moe.json' for GPT_MoE model. Please
carefully check the model settings.

## Run Guide
First, start TePDist server side, then:

> run ./run_gpt.sh to do test with fake inputs with GPT model.
> run ./run_moe.sh to do test with fake inputs with GPT_MoE model.

## TO BE NOTICED:
- For GPT model settings,
    - intermediate_size = 4 * hidden_size
- For GShard and GSPMD MoE settings,
    - intermediate_size = 8 * hidden_size
    - 'num_local_groups (G)' equals to
          'batch_size * seq_length / expert_group_size (S)'.
      Here we need to decide G and C, S would be determined. It means that
      when 'batch_size' is enlarged, do same on G.
    - moe_gating is better to set as 'top_2', which is different with M6.
      Due to 'top_2' setting, expert capacity would be directly decided by
      'expert_capacity_dim (C)'. Ferthermore, 'second_expert_policy' and
      'second_expert_threshold' should be set but would not influent the
      computation complexity of MoE model.
