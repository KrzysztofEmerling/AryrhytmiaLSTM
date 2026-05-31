from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization
)

def build_lstm():
    model = Sequential([
        Input(shape=(200, 1)),
        LSTM(32),
        Dense(32, activation='relu'),
        Dropout(0.2),

        Dense(5, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_lstm_from_hp(hp):

    model = Sequential()

    n_lstm = hp.Int(
        "n_lstm_layers",
        min_value=1,
        max_value=3
    )

    for i in range(n_lstm):

        units = hp.Int(
            f"lstm_units_{i}",
            min_value=4,
            max_value=7
        )

        units = 2 ** units

        dropout_rate = hp.Float(
            f"lstm_dropout_{i}",
            min_value=0.0,
            max_value=0.5,
            step=0.1
        )

        recurrent_dropout = hp.Float(
            f"recurrent_dropout_{i}",
            min_value=0.0,
            max_value=0.5,
            step=0.1
        )

        reg_type = hp.Choice(
            f"lstm_reg_type_{i}",
            values=["none", "l1", "l2", "l1_l2"]
        )

        reg_strength = hp.Float(
            f"lstm_reg_strength_{i}",
            min_value=1e-6,
            max_value=1e-2,
            sampling="log"
        )

        if reg_type == "l1":
            reg = l1(reg_strength)

        elif reg_type == "l2":
            reg = l2(reg_strength)

        elif reg_type == "l1_l2":
            reg = l1_l2(reg_strength)

        else:
            reg = None

        return_sequences = i < (n_lstm - 1)

        if i == 0:

            model.add(
                LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=dropout_rate,
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=reg,
                    input_shape=(200, 1)
                )
            )

        else:

            model.add(
                LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=dropout_rate,
                    recurrent_dropout=recurrent_dropout,
                    kernel_regularizer=reg
                )
            )

        model.add(BatchNormalization())

    n_dense = hp.Int(
        "n_dense_layers",
        min_value=1,
        max_value=3
    )

    for i in range(n_dense):

        units = hp.Int(
            f"dense_units_{i}",
            min_value=4,
            max_value=7
        )

        units = 2 ** units

        dropout_rate = hp.Float(
            f"dense_dropout_{i}",
            min_value=0.0,
            max_value=0.5,
            step=0.1
        )

        reg_type = hp.Choice(
            f"dense_reg_type_{i}",
            values=["none", "l1", "l2", "l1_l2"]
        )

        reg_strength = hp.Float(
            f"dense_reg_strength_{i}",
            min_value=1e-6,
            max_value=1e-2,
            sampling="log"
        )

        if reg_type == "l1":
            reg = l1(reg_strength)

        elif reg_type == "l2":
            reg = l2(reg_strength)

        elif reg_type == "l1_l2":
            reg = l1_l2(reg_strength)

        else:
            reg = None

        model.add(
            Dense(
                units,
                activation='relu',
                kernel_regularizer=reg
            )
        )

        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(Dense(5, activation='softmax'))

    model.compile(
        optimizer=Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model