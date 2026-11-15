// Restormer, fine-tuned per HDC level (Phase 2, Levels 0–9). Generated from
// "Restormer Fine Tuned/report/tables/*.csv" (all_levels_summary_extended, all_levels_by_font,
// gains_fine_tuned, training_and_registration_by_level). Test set: 40 pages per level.
// Level 0 has no OCR; its PSNR/SSIM were recomputed from the saved outputs. Level 5 is the
// legacy run: its date, sigma and patch/batch are inferred (train.inferred).
// psnr in dB; ssim 0–1; char, token, cer, wer, exact, empty in %.
// wins: test images (of 40) where fine-tuned beats the blurred input; char = [win, tie, loss].
export const restormerLevels = [
  {
    "level": 0,
    "input": {
      "psnr": 26.08,
      "ssim": 0.5236
    },
    "pre": {
      "psnr": 24.98,
      "ssim": 0.5311
    },
    "ft": {
      "psnr": 28.74,
      "ssim": 0.6908
    },
    "wins": {
      "psnr": 40,
      "ssim": 40
    },
    "train": {
      "date": "2026-11-12",
      "sigma": 5,
      "patch": 128,
      "batch": 4,
      "nccBefore": 0.897,
      "nccAfter": 0.949,
      "nccWorst": 0.938,
      "valPsnr0": 24.71,
      "valSsim0": 0.5182,
      "bestIter": 3000,
      "bestValPsnr": 28.17,
      "bestValSsim": 0.68,
      "lossFirst": 0.0376,
      "lossLast": 0.0279,
      "minutes": 49,
      "scaleX": 0.9934,
      "scaleY": 0.9882,
      "tx": 7.1,
      "ty": 13.4
    },
    "byFont": {
      "input": {
        "timesR": {
          "psnr": 27.11,
          "ssim": 0.5663
        },
        "verdanaRef": {
          "psnr": 25.05,
          "ssim": 0.4809
        }
      },
      "pre": {
        "timesR": {
          "psnr": 25.58,
          "ssim": 0.5606
        },
        "verdanaRef": {
          "psnr": 24.39,
          "ssim": 0.5017
        }
      },
      "ft": {
        "timesR": {
          "psnr": 30.41,
          "ssim": 0.718
        },
        "verdanaRef": {
          "psnr": 27.06,
          "ssim": 0.6635
        }
      }
    }
  },
  {
    "level": 1,
    "input": {
      "psnr": 23.17,
      "ssim": 0.4193,
      "char": 98.35,
      "token": 85.79,
      "cer": 1.65,
      "wer": 15.88,
      "exact": 55.0,
      "empty": 0.0
    },
    "pre": {
      "psnr": 22.47,
      "ssim": 0.4552,
      "char": 98.11,
      "token": 83.71,
      "cer": 1.89,
      "wer": 19.62,
      "exact": 50.0,
      "empty": 0.0
    },
    "ft": {
      "psnr": 28.87,
      "ssim": 0.6265,
      "char": 92.62,
      "token": 60.5,
      "cer": 7.38,
      "wer": 56.88,
      "exact": 20.0,
      "empty": 0.0
    },
    "gt": {
      "char": 90.81,
      "token": 57.42,
      "cer": 9.19,
      "wer": 64.96,
      "exact": 15.0,
      "empty": 0.0
    },
    "wins": {
      "psnr": 40,
      "ssim": 40,
      "char": [
        1,
        11,
        28
      ]
    },
    "train": {
      "date": "2026-11-12",
      "sigma": 5,
      "patch": 128,
      "batch": 4,
      "nccBefore": 0.95,
      "nccAfter": 0.97,
      "nccWorst": 0.966,
      "valPsnr0": 22.46,
      "valSsim0": 0.4579,
      "bestIter": 2500,
      "bestValPsnr": 28.83,
      "bestValSsim": 0.6277,
      "lossFirst": 0.0468,
      "lossLast": 0.0271,
      "minutes": 45,
      "scaleX": 1.0003,
      "scaleY": 0.9951,
      "tx": -0.8,
      "ty": 10.3
    },
    "byFont": {
      "input": {
        "timesR": {
          "psnr": 23.12,
          "ssim": 0.4251,
          "char": 98.44,
          "token": 86.17
        },
        "verdanaRef": {
          "psnr": 23.22,
          "ssim": 0.4136,
          "char": 98.27,
          "token": 85.42
        }
      },
      "pre": {
        "timesR": {
          "psnr": 22.39,
          "ssim": 0.46,
          "char": 97.96,
          "token": 82.0
        },
        "verdanaRef": {
          "psnr": 22.54,
          "ssim": 0.4504,
          "char": 98.27,
          "token": 85.42
        }
      },
      "ft": {
        "timesR": {
          "psnr": 29.54,
          "ssim": 0.6327,
          "char": 89.49,
          "token": 51.75
        },
        "verdanaRef": {
          "psnr": 28.2,
          "ssim": 0.6202,
          "char": 95.75,
          "token": 69.25
        }
      },
      "gt": {
        "timesR": {
          "char": 87.27,
          "token": 45.58
        },
        "verdanaRef": {
          "char": 94.35,
          "token": 69.25
        }
      }
    }
  },
  {
    "level": 2,
    "input": {
      "psnr": 19.37,
      "ssim": 0.3902,
      "char": 94.25,
      "token": 63.54,
      "cer": 5.75,
      "wer": 42.71,
      "exact": 22.5,
      "empty": 0.0
    },
    "pre": {
      "psnr": 18.84,
      "ssim": 0.4403,
      "char": 94.97,
      "token": 65.0,
      "cer": 5.03,
      "wer": 39.79,
      "exact": 25.0,
      "empty": 0.0
    },
    "ft": {
      "psnr": 27.38,
      "ssim": 0.6151,
      "char": 96.07,
      "token": 75.0,
      "cer": 3.93,
      "wer": 27.5,
      "exact": 37.5,
      "empty": 0.0
    },
    "gt": {
      "char": 92.16,
      "token": 55.62,
      "cer": 7.84,
      "wer": 58.54,
      "exact": 20.0,
      "empty": 0.0
    },
    "wins": {
      "psnr": 40,
      "ssim": 40,
      "char": [
        18,
        17,
        5
      ]
    },
    "train": {
      "date": "2026-11-12",
      "sigma": 5,
      "patch": 128,
      "batch": 4,
      "nccBefore": 0.906,
      "nccAfter": 0.96,
      "nccWorst": 0.953,
      "valPsnr0": 18.93,
      "valSsim0": 0.4393,
      "bestIter": 3000,
      "bestValPsnr": 27.47,
      "bestValSsim": 0.6129,
      "lossFirst": 0.0661,
      "lossLast": 0.0307,
      "minutes": 43,
      "scaleX": 1.0072,
      "scaleY": 1.0029,
      "tx": -7.8,
      "ty": 6.4
    },
    "byFont": {
      "input": {
        "timesR": {
          "psnr": 19.21,
          "ssim": 0.3943,
          "char": 90.53,
          "token": 45.83
        },
        "verdanaRef": {
          "psnr": 19.52,
          "ssim": 0.3861,
          "char": 97.97,
          "token": 81.25
        }
      },
      "pre": {
        "timesR": {
          "psnr": 18.73,
          "ssim": 0.445,
          "char": 91.81,
          "token": 47.5
        },
        "verdanaRef": {
          "psnr": 18.96,
          "ssim": 0.4356,
          "char": 98.12,
          "token": 82.5
        }
      },
      "ft": {
        "timesR": {
          "psnr": 27.9,
          "ssim": 0.6214,
          "char": 93.85,
          "token": 65.83
        },
        "verdanaRef": {
          "psnr": 26.86,
          "ssim": 0.6089,
          "char": 98.28,
          "token": 84.17
        }
      },
      "gt": {
        "timesR": {
          "char": 88.69,
          "token": 42.5
        },
        "verdanaRef": {
          "char": 95.62,
          "token": 68.75
        }
      }
    }
  },
  {
    "level": 3,
    "input": {
      "psnr": 17.34,
      "ssim": 0.3795,
      "char": 75.55,
      "token": 24.04,
      "cer": 24.45,
      "wer": 88.04,
      "exact": 2.5,
      "empty": 0.0
    },
    "pre": {
      "psnr": 16.95,
      "ssim": 0.4353,
      "char": 75.54,
      "token": 23.5,
      "cer": 24.46,
      "wer": 93.29,
      "exact": 2.5,
      "empty": 0.0
    },
    "ft": {
      "psnr": 25.72,
      "ssim": 0.6056,
      "char": 98.28,
      "token": 86.46,
      "cer": 1.72,
      "wer": 14.17,
      "exact": 62.5,
      "empty": 0.0
    },
    "gt": {
      "char": 92.48,
      "token": 63.79,
      "cer": 7.52,
      "wer": 49.92,
      "exact": 20.0,
      "empty": 0.0
    },
    "wins": {
      "psnr": 40,
      "ssim": 40,
      "char": [
        38,
        2,
        0
      ]
    },
    "train": {
      "date": "2026-11-12",
      "sigma": 12,
      "patch": 128,
      "batch": 4,
      "nccBefore": 0.864,
      "nccAfter": 0.948,
      "nccWorst": 0.941,
      "valPsnr0": 17.03,
      "valSsim0": 0.4375,
      "bestIter": 2000,
      "bestValPsnr": 25.76,
      "bestValSsim": 0.6052,
      "lossFirst": 0.0966,
      "lossLast": 0.0355,
      "minutes": 48,
      "scaleX": 1.0143,
      "scaleY": 1.0097,
      "tx": -14.8,
      "ty": 3.8
    },
    "byFont": {
      "input": {
        "timesR": {
          "psnr": 17.39,
          "ssim": 0.3831,
          "char": 60.18,
          "token": 3.0
        },
        "verdanaRef": {
          "psnr": 17.28,
          "ssim": 0.3759,
          "char": 90.92,
          "token": 45.08
        }
      },
      "pre": {
        "timesR": {
          "psnr": 17.05,
          "ssim": 0.4382,
          "char": 60.64,
          "token": 6.5
        },
        "verdanaRef": {
          "psnr": 16.86,
          "ssim": 0.4324,
          "char": 90.44,
          "token": 40.5
        }
      },
      "ft": {
        "timesR": {
          "psnr": 25.71,
          "ssim": 0.6092,
          "char": 99.06,
          "token": 92.75
        },
        "verdanaRef": {
          "psnr": 25.73,
          "ssim": 0.602,
          "char": 97.49,
          "token": 80.17
        }
      },
      "gt": {
        "timesR": {
          "char": 89.8,
          "token": 55.58
        },
        "verdanaRef": {
          "char": 95.15,
          "token": 72.0
        }
      }
    }
  },
  {
    "level": 4,
    "input": {
      "psnr": 16.07,
      "ssim": 0.373,
      "char": 23.48,
      "token": 0.5,
      "cer": 76.52,
      "wer": 130.17,
      "exact": 0.0,
      "empty": 0.0
    },
    "pre": {
      "psnr": 15.86,
      "ssim": 0.4241,
      "char": 27.34,
      "token": 0.5,
      "cer": 72.66,
      "wer": 134.96,
      "exact": 0.0,
      "empty": 0.0
    },
    "ft": {
      "psnr": 24.1,
      "ssim": 0.5949,
      "char": 97.4,
      "token": 82.17,
      "cer": 2.6,
      "wer": 17.83,
      "exact": 47.5,
      "empty": 0.0
    },
    "gt": {
      "char": 91.27,
      "token": 55.12,
      "cer": 8.73,
      "wer": 65.88,
      "exact": 15.0,
      "empty": 0.0
    },
    "wins": {
      "psnr": 40,
      "ssim": 40,
      "char": [
        40,
        0,
        0
      ]
    },
    "train": {
      "date": "2026-11-12",
      "sigma": 20,
      "patch": 128,
      "batch": 4,
      "nccBefore": 0.824,
      "nccAfter": 0.915,
      "nccWorst": 0.905,
      "valPsnr0": 15.85,
      "valSsim0": 0.4249,
      "bestIter": 3000,
      "bestValPsnr": 23.95,
      "bestValSsim": 0.5939,
      "lossFirst": 0.129,
      "lossLast": 0.0417,
      "minutes": 42,
      "scaleX": 1.0217,
      "scaleY": 1.0177,
      "tx": -23.1,
      "ty": 0.8
    },
    "byFont": {
      "input": {
        "timesR": {
          "psnr": 16.3,
          "ssim": 0.3778,
          "char": 14.59,
          "token": 0.0
        },
        "verdanaRef": {
          "psnr": 15.83,
          "ssim": 0.3682,
          "char": 32.36,
          "token": 1.0
        }
      },
      "pre": {
        "timesR": {
          "psnr": 16.11,
          "ssim": 0.4286,
          "char": 18.07,
          "token": 0.0
        },
        "verdanaRef": {
          "psnr": 15.61,
          "ssim": 0.4196,
          "char": 36.6,
          "token": 1.0
        }
      },
      "ft": {
        "timesR": {
          "psnr": 23.92,
          "ssim": 0.5962,
          "char": 96.86,
          "token": 79.67
        },
        "verdanaRef": {
          "psnr": 24.28,
          "ssim": 0.5937,
          "char": 97.93,
          "token": 84.67
        }
      },
      "gt": {
        "timesR": {
          "char": 88.54,
          "token": 44.42
        },
        "verdanaRef": {
          "char": 93.99,
          "token": 65.83
        }
      }
    }
  },
  {
    "level": 5,
    "input": {
      "psnr": 15.35,
      "ssim": 0.3724,
      "char": 0.0,
      "token": 0.0,
      "cer": 100.0,
      "wer": 100.0,
      "exact": 0.0,
      "empty": 100.0
    },
    "pre": {
      "psnr": 15.29,
      "ssim": 0.4188,
      "char": 0.0,
      "token": 0.0,
      "cer": 100.0,
      "wer": 100.0,
      "exact": 0.0,
      "empty": 100.0
    },
    "ft": {
      "psnr": 21.98,
      "ssim": 0.5824,
      "char": 96.94,
      "token": 75.0,
      "cer": 3.06,
      "wer": 25.83,
      "exact": 40.0,
      "empty": 0.0
    },
    "gt": {
      "char": 94.11,
      "token": 64.71,
      "cer": 5.89,
      "wer": 51.54,
      "exact": 25.0,
      "empty": 0.0
    },
    "wins": {
      "psnr": 40,
      "ssim": 40,
      "char": [
        40,
        0,
        0
      ]
    },
    "train": {
      "nccBefore": 0.792,
      "nccAfter": 0.892,
      "nccWorst": 0.878,
      "valPsnr0": 15.38,
      "valSsim0": 0.4225,
      "bestIter": 3000,
      "bestValPsnr": 22.01,
      "bestValSsim": 0.5832,
      "lossFirst": 0.1416,
      "lossLast": 0.0557,
      "scaleX": 1.0295,
      "scaleY": 1.0245,
      "tx": -32.1,
      "ty": 0.4,
      "date": "2026-10",
      "sigma": 25,
      "patch": 128,
      "batch": 4,
      "inferred": true
    },
    "byFont": {
      "input": {
        "timesR": {
          "psnr": 15.58,
          "ssim": 0.378,
          "char": 0.0,
          "token": 0.0
        },
        "verdanaRef": {
          "psnr": 15.12,
          "ssim": 0.3669,
          "char": 0.0,
          "token": 0.0
        }
      },
      "pre": {
        "timesR": {
          "psnr": 15.53,
          "ssim": 0.4243,
          "char": 0.0,
          "token": 0.0
        },
        "verdanaRef": {
          "psnr": 15.04,
          "ssim": 0.4133,
          "char": 0.0,
          "token": 0.0
        }
      },
      "ft": {
        "timesR": {
          "psnr": 21.77,
          "ssim": 0.5839,
          "char": 96.71,
          "token": 73.25
        },
        "verdanaRef": {
          "psnr": 22.18,
          "ssim": 0.5809,
          "char": 97.16,
          "token": 76.75
        }
      },
      "gt": {
        "timesR": {
          "char": 92.15,
          "token": 58.08
        },
        "verdanaRef": {
          "char": 96.07,
          "token": 71.33
        }
      }
    }
  },
  {
    "level": 6,
    "input": {
      "psnr": 15.03,
      "ssim": 0.3723,
      "char": 0.0,
      "token": 0.0,
      "cer": 100.0,
      "wer": 100.0,
      "exact": 0.0,
      "empty": 100.0
    },
    "pre": {
      "psnr": 15.05,
      "ssim": 0.4174,
      "char": 0.0,
      "token": 0.0,
      "cer": 100.0,
      "wer": 100.0,
      "exact": 0.0,
      "empty": 100.0
    },
    "ft": {
      "psnr": 20.19,
      "ssim": 0.5693,
      "char": 83.22,
      "token": 32.0,
      "cer": 16.78,
      "wer": 78.83,
      "exact": 0.0,
      "empty": 0.0
    },
    "gt": {
      "char": 93.41,
      "token": 69.12,
      "cer": 6.59,
      "wer": 38.67,
      "exact": 20.0,
      "empty": 0.0
    },
    "wins": {
      "psnr": 40,
      "ssim": 40,
      "char": [
        40,
        0,
        0
      ]
    },
    "train": {
      "date": "2026-10-19",
      "sigma": 30,
      "patch": 128,
      "batch": 4,
      "nccBefore": 0.758,
      "nccAfter": 0.867,
      "nccWorst": 0.849,
      "valPsnr0": 15.03,
      "valSsim0": 0.4172,
      "bestIter": 3000,
      "bestValPsnr": 20.19,
      "bestValSsim": 0.5684,
      "lossFirst": 0.1499,
      "lossLast": 0.0708,
      "minutes": 44,
      "scaleX": 1.0359,
      "scaleY": 1.0362,
      "tx": -37.1,
      "ty": -8.8
    },
    "byFont": {
      "input": {
        "timesR": {
          "psnr": 15.32,
          "ssim": 0.3766,
          "char": 0.0,
          "token": 0.0
        },
        "verdanaRef": {
          "psnr": 14.75,
          "ssim": 0.368,
          "char": 0.0,
          "token": 0.0
        }
      },
      "pre": {
        "timesR": {
          "psnr": 15.35,
          "ssim": 0.422,
          "char": 0.0,
          "token": 0.0
        },
        "verdanaRef": {
          "psnr": 14.75,
          "ssim": 0.4127,
          "char": 0.0,
          "token": 0.0
        }
      },
      "ft": {
        "timesR": {
          "psnr": 19.83,
          "ssim": 0.5683,
          "char": 78.52,
          "token": 23.92
        },
        "verdanaRef": {
          "psnr": 20.55,
          "ssim": 0.5703,
          "char": 87.91,
          "token": 40.08
        }
      },
      "gt": {
        "timesR": {
          "char": 91.69,
          "token": 66.42
        },
        "verdanaRef": {
          "char": 95.12,
          "token": 71.83
        }
      }
    }
  },
  {
    "level": 7,
    "input": {
      "psnr": 14.71,
      "ssim": 0.3729,
      "char": 0.0,
      "token": 0.0,
      "cer": 100.0,
      "wer": 100.0,
      "exact": 0.0,
      "empty": 100.0
    },
    "pre": {
      "psnr": 14.76,
      "ssim": 0.4171,
      "char": 0.0,
      "token": 0.0,
      "cer": 100.0,
      "wer": 100.0,
      "exact": 0.0,
      "empty": 100.0
    },
    "ft": {
      "psnr": 19.87,
      "ssim": 0.5665,
      "char": 77.04,
      "token": 22.29,
      "cer": 22.96,
      "wer": 98.12,
      "exact": 0.0,
      "empty": 0.0
    },
    "gt": {
      "char": 91.63,
      "token": 61.67,
      "cer": 8.37,
      "wer": 61.25,
      "exact": 20.0,
      "empty": 0.0
    },
    "wins": {
      "psnr": 40,
      "ssim": 40,
      "char": [
        40,
        0,
        0
      ]
    },
    "train": {
      "date": "2026-11-11",
      "sigma": 37,
      "patch": 192,
      "batch": 2,
      "nccBefore": 0.722,
      "nccAfter": 0.836,
      "nccWorst": 0.805,
      "valPsnr0": 14.8,
      "valSsim0": 0.415,
      "bestIter": 2500,
      "bestValPsnr": 20.02,
      "bestValSsim": 0.5667,
      "lossFirst": 0.1401,
      "lossLast": 0.0634,
      "minutes": 81,
      "scaleX": 1.043,
      "scaleY": 1.0373,
      "tx": -48.4,
      "ty": -2.9
    },
    "byFont": {
      "input": {
        "timesR": {
          "psnr": 14.93,
          "ssim": 0.3751,
          "char": 0.0,
          "token": 0.0
        },
        "verdanaRef": {
          "psnr": 14.5,
          "ssim": 0.3707,
          "char": 0.0,
          "token": 0.0
        }
      },
      "pre": {
        "timesR": {
          "psnr": 14.99,
          "ssim": 0.4192,
          "char": 0.0,
          "token": 0.0
        },
        "verdanaRef": {
          "psnr": 14.54,
          "ssim": 0.4149,
          "char": 0.0,
          "token": 0.0
        }
      },
      "ft": {
        "timesR": {
          "psnr": 19.33,
          "ssim": 0.5631,
          "char": 66.34,
          "token": 5.42
        },
        "verdanaRef": {
          "psnr": 20.42,
          "ssim": 0.5699,
          "char": 87.74,
          "token": 39.17
        }
      },
      "gt": {
        "timesR": {
          "char": 87.95,
          "token": 53.33
        },
        "verdanaRef": {
          "char": 95.3,
          "token": 70.0
        }
      }
    }
  },
  {
    "level": 8,
    "input": {
      "psnr": 14.45,
      "ssim": 0.4173,
      "char": 0.0,
      "token": 0.0,
      "cer": 100.0,
      "wer": 100.0,
      "exact": 0.0,
      "empty": 100.0
    },
    "pre": {
      "psnr": 14.44,
      "ssim": 0.4579,
      "char": 0.0,
      "token": 0.0,
      "cer": 100.0,
      "wer": 100.0,
      "exact": 0.0,
      "empty": 100.0
    },
    "ft": {
      "psnr": 19.47,
      "ssim": 0.5629,
      "char": 73.47,
      "token": 23.92,
      "cer": 26.53,
      "wer": 98.38,
      "exact": 0.0,
      "empty": 0.0
    },
    "gt": {
      "char": 92.62,
      "token": 63.83,
      "cer": 7.38,
      "wer": 56.5,
      "exact": 17.5,
      "empty": 0.0
    },
    "wins": {
      "psnr": 40,
      "ssim": 40,
      "char": [
        40,
        0,
        0
      ]
    },
    "train": {
      "date": "2026-11-11",
      "sigma": 45,
      "patch": 192,
      "batch": 2,
      "nccBefore": 0.669,
      "nccAfter": 0.804,
      "nccWorst": 0.755,
      "valPsnr0": 14.43,
      "valSsim0": 0.4577,
      "bestIter": 3000,
      "bestValPsnr": 19.48,
      "bestValSsim": 0.5635,
      "lossFirst": 0.1429,
      "lossLast": 0.0748,
      "minutes": 88,
      "scaleX": 1.0524,
      "scaleY": 1.0443,
      "tx": -55.9,
      "ty": -4.5
    },
    "byFont": {
      "input": {
        "timesR": {
          "psnr": 14.66,
          "ssim": 0.4199,
          "char": 0.0,
          "token": 0.0
        },
        "verdanaRef": {
          "psnr": 14.24,
          "ssim": 0.4147,
          "char": 0.0,
          "token": 0.0
        }
      },
      "pre": {
        "timesR": {
          "psnr": 14.65,
          "ssim": 0.4604,
          "char": 0.0,
          "token": 0.0
        },
        "verdanaRef": {
          "psnr": 14.23,
          "ssim": 0.4554,
          "char": 0.0,
          "token": 0.0
        }
      },
      "ft": {
        "timesR": {
          "psnr": 18.75,
          "ssim": 0.5589,
          "char": 57.47,
          "token": 8.5
        },
        "verdanaRef": {
          "psnr": 20.19,
          "ssim": 0.5669,
          "char": 89.48,
          "token": 39.33
        }
      },
      "gt": {
        "timesR": {
          "char": 91.37,
          "token": 60.17
        },
        "verdanaRef": {
          "char": 93.88,
          "token": 67.5
        }
      }
    }
  },
  {
    "level": 9,
    "input": {
      "psnr": 14.39,
      "ssim": 0.4192,
      "char": 0.0,
      "token": 0.0,
      "cer": 100.0,
      "wer": 100.0,
      "exact": 0.0,
      "empty": 100.0
    },
    "pre": {
      "psnr": 14.38,
      "ssim": 0.4595,
      "char": 0.0,
      "token": 0.0,
      "cer": 100.0,
      "wer": 100.0,
      "exact": 0.0,
      "empty": 100.0
    },
    "ft": {
      "psnr": 16.81,
      "ssim": 0.531,
      "char": 52.82,
      "token": 10.87,
      "cer": 47.18,
      "wer": 133.12,
      "exact": 0.0,
      "empty": 0.0
    },
    "gt": {
      "char": 93.24,
      "token": 66.79,
      "cer": 6.76,
      "wer": 50.17,
      "exact": 32.5,
      "empty": 0.0
    },
    "wins": {
      "psnr": 40,
      "ssim": 40,
      "char": [
        40,
        0,
        0
      ]
    },
    "train": {
      "date": "2026-11-11",
      "sigma": 47,
      "patch": 192,
      "batch": 2,
      "nccBefore": 0.626,
      "nccAfter": 0.78,
      "nccWorst": 0.73,
      "valPsnr0": 14.38,
      "valSsim0": 0.4583,
      "bestIter": 2000,
      "bestValPsnr": 16.7,
      "bestValSsim": 0.5286,
      "lossFirst": 0.1377,
      "lossLast": 0.0845,
      "minutes": 85,
      "scaleX": 1.0573,
      "scaleY": 1.0524,
      "tx": -68.3,
      "ty": -8.5
    },
    "byFont": {
      "input": {
        "timesR": {
          "psnr": 14.6,
          "ssim": 0.4216,
          "char": 0.0,
          "token": 0.0
        },
        "verdanaRef": {
          "psnr": 14.17,
          "ssim": 0.4168,
          "char": 0.0,
          "token": 0.0
        }
      },
      "pre": {
        "timesR": {
          "psnr": 14.59,
          "ssim": 0.4619,
          "char": 0.0,
          "token": 0.0
        },
        "verdanaRef": {
          "psnr": 14.17,
          "ssim": 0.4571,
          "char": 0.0,
          "token": 0.0
        }
      },
      "ft": {
        "timesR": {
          "psnr": 16.55,
          "ssim": 0.5299,
          "char": 34.33,
          "token": 3.25
        },
        "verdanaRef": {
          "psnr": 17.06,
          "ssim": 0.532,
          "char": 71.3,
          "token": 18.5
        }
      },
      "gt": {
        "timesR": {
          "char": 90.26,
          "token": 54.08
        },
        "verdanaRef": {
          "char": 96.22,
          "token": 79.5
        }
      }
    }
  }
]
